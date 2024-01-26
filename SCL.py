import numpy as np
import matplotlib.pyplot as plt

def competitive_learning(data, num_neurons, learning_rate=0.1, epochs=100):
    # Initialize random weights for neurons
    weights = np.random.rand(num_neurons, data.shape[1])

    for epoch in range(epochs):
        # Shuffle the data for each epoch
        np.random.shuffle(data)

        for input_vector in data:
            # Compute the winner neuron (index) based on the minimum Euclidean distance
            winner_index = np.argmin(np.linalg.norm(weights - input_vector, axis=1))

            # Update the weights of the winner neuron
            weights[winner_index] += learning_rate * (input_vector - weights[winner_index])

    return weights

# Generate random 2D data
np.random.seed(42)
data = np.random.rand(100, 2)

# Apply competitive learning
num_neurons = 5 # represents centroids
learned_weights = competitive_learning(data, num_neurons)

# Plot the data and learned neuron weights
plt.scatter(data[:, 0], data[:, 1], label='Data')
plt.scatter(learned_weights[:, 0], learned_weights[:, 1], marker='X', s=200, c='red', label='Learned Neurons')
plt.title('Competitive Learning')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()