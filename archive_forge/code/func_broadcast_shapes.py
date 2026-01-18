import math
import numpy as np
import tree
from keras.src.api_export import keras_export
def broadcast_shapes(shape1, shape2):
    """Broadcast input shapes to a unified shape.

    Convert to list for mutability.

    Args:
        shape1: A tuple or list of integers.
        shape2: A tuple or list of integers.

    Returns:
        output_shape (list of integers or `None`): The broadcasted shape.

    Example:
    >>> broadcast_shapes((5, 3), (1, 3))
    [5, 3]
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    origin_shape1 = shape1
    origin_shape2 = shape2
    if len(shape1) > len(shape2):
        shape2 = [1] * (len(shape1) - len(shape2)) + shape2
    if len(shape1) < len(shape2):
        shape1 = [1] * (len(shape2) - len(shape1)) + shape1
    output_shape = list(shape1)
    for i in range(len(shape1)):
        if shape1[i] == 1:
            output_shape[i] = shape2[i]
        elif shape1[i] is None:
            output_shape[i] = None if shape2[i] == 1 else shape2[i]
        elif shape2[i] == 1 or shape2[i] is None or shape2[i] == shape1[i]:
            output_shape[i] = shape1[i]
        else:
            raise ValueError(f'Cannot broadcast shape, the failure dim has value {shape1[i]}, which cannot be broadcasted to {shape2[i]}. Input shapes are: {origin_shape1} and {origin_shape2}.')
    return output_shape