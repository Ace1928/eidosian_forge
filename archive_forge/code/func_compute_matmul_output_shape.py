import math
import numpy as np
import tree
from keras.src.api_export import keras_export
def compute_matmul_output_shape(shape1, shape2):
    """Compute the output shape of a `matmul` operation.

    Args:
        shape1: Shape of the left operand.
        shape2: Shape of the right operand.

    Returns:
        Tuple of ints: The output shape for the `matmul` operation.
    """
    if len(shape1) == 1:
        shape1 = (1, shape1[0])
    if len(shape2) == 1:
        shape2 = (shape2[0], 1)
    if shape1[-1] is not None and shape2[-2] is not None and (shape1[-1] != shape2[-2]):
        raise ValueError(f'Inner dimensions (`x1.shape[-1]` and `x2.shape[-2]`) must be equal, but received `x1.shape={shape1}` and `x2.shape={shape2}`.')
    leading_shape = broadcast_shapes(shape1[:-2], shape2[:-2])
    last_2_dims_shape = [shape1[-2], shape2[-1]]
    output_shape = leading_shape + last_2_dims_shape
    if len(shape1) == 1:
        del output_shape[-2]
    if len(shape2) == 1:
        del output_shape[-1]
    return tuple(output_shape)