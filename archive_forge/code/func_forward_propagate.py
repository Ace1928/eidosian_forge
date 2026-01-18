import numpy as np
from config import global_config
from logger import CustomLogger
def forward_propagate(self, input_data):
    """
        Performs forward propagation through the neural network.

        Parameters:
            input_data (np.array): The input data to the neural network.

        Returns:
            np.array: The output of the neural network.
        """
    current_layer_output = input_data
    for layer_index, (layer_name, layer_info) in enumerate(self.network_structure.items(), start=1):
        if layer_name in self.weights:
            current_layer_output = np.dot(current_layer_output, self.weights[layer_name])
            vectorized_activation = np.vectorize(lambda x: self.activate(x, layer_info['Activation_Function']))
            current_layer_output = vectorized_activation(current_layer_output)
            self.logger.debug(f'Layer {layer_index} output: {current_layer_output}')
    return current_layer_output