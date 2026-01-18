import math
import numpy as np
import tree
from keras.src.api_export import keras_export
def compute_expand_dims_output_shape(input_shape, axis):
    """Compute the output shape for the `expand_dims` operation.

    Args:
        input_shape: Input shape.
        axis: int for the axis to expand.

    Returns:
        Tuple of ints: The output shape after the `expand_dims` operation.
    """
    input_shape = list(input_shape)
    if axis is None:
        axis = len(input_shape)
    elif axis < 0:
        axis = len(input_shape) + 1 + axis
    return tuple(input_shape[:axis] + [1] + input_shape[axis:])