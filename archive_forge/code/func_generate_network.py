import math
from config import global_config
def generate_network(self):
    """
        Generates the neural network based on the calculated layers.

        Returns:
            dict: A dictionary representing the neural network with layers and hexagons.
        """
    network = {}
    for layer_index, hexagons in enumerate(self.layers):
        network[f'Layer_{layer_index + 1}'] = {'Hexagons': hexagons, 'Activation_Function': global_config.get_activation_function(global_config.default_activation_function)}
    return network