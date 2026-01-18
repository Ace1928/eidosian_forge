from typing import List, Tuple, Dict
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def generate_3d_structure(layers: int, side_length: float) -> Dict[int, List[Hexagon3D]]:
    """
    Generate a 3D structure of stacked hexagons, where each layer above is derived from
    the central points of the hexagons in the layer below. This structure forms a pyramidal
    shape made of hexagons.

    The function iterates through each layer, starting from the base layer, generating
    hexagons positioned based on the hexagons from the previous layer, progressively
    increasing the elevation for each layer.

    Args:
        layers (int): The number of layers to generate, including the base layer.
        side_length (float): The side length of each hexagon in the structure.

    Returns:
        Dict[int, List[Hexagon3D]]: A dictionary mapping each layer index to a list of
                                    hexagons (defined by their vertices) in that layer.
    """
    structure = {}
    elevation = 0.0
    elevation_step = side_length * math.sqrt(3) / 2
    base_center = (0.0, 0.0, elevation)
    structure[0] = [generate_hexagon(base_center, side_length, elevation)]
    for layer in range(1, layers):
        elevation += elevation_step
        previous_layer_hexagons = structure[layer - 1]
        current_layer_hexagons = []
        for hexagon in previous_layer_hexagons:
            for i in range(6):
                angle_rad = math.pi / 3 * i
                x = hexagon[0][0] + side_length * 2 * math.cos(angle_rad)
                y = hexagon[0][1] + side_length * 2 * math.sin(angle_rad)
                new_hexagon_center = (x, y, elevation)
                if not any((np.allclose(new_hexagon_center, hex[0]) for hex in current_layer_hexagons)):
                    current_layer_hexagons.append(generate_hexagon(new_hexagon_center, side_length, elevation))
        structure[layer] = current_layer_hexagons
    return structure