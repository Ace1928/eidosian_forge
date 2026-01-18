import math
import numpy as np
import tree
from keras.src.api_export import keras_export
def compute_pooling_output_shape(input_shape, pool_size, strides, padding='valid', data_format='channels_last'):
    """Computes the output shape of pooling operations.

    Args:
        input_shape: Input shape. Must be a tuple of integers.
        pool_size: Size of the pooling operation. Must be a tuple of integers.
        strides: Stride of the pooling operation. Must be a tuple of integers.
            Defaults to `pool_size`.
        padding: Padding method. Available methods are `"valid"` or `"same"`.
            Defaults to `"valid"`.
        data_format: String, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, weight)`. Defaults to `"channels_last"`.

    Returns:
        Tuple of ints: The output shape of the pooling operation.

    Examples:

    # Basic usage with square pooling on a single image
    >>> compute_pooling_output_shape((1, 4, 4, 1), (2, 2))
    (1, 2, 2, 1)

    # Strided pooling on a single image with strides different from pool_size
    >>> compute_pooling_output_shape((1, 4, 4, 1), (2, 2), strides=(1, 1))
    (1, 3, 3, 1)

    # Pooling on a batch of images
    >>> compute_pooling_output_shape((32, 4, 4, 3), (2, 2))
    (32, 2, 2, 3)
    """
    strides = pool_size if strides is None else strides
    input_shape_origin = list(input_shape)
    input_shape = np.array(input_shape)
    if data_format == 'channels_last':
        spatial_shape = input_shape[1:-1]
    else:
        spatial_shape = input_shape[2:]
    none_dims = []
    for i in range(len(spatial_shape)):
        if spatial_shape[i] is None:
            spatial_shape[i] = -1
            none_dims.append(i)
    pool_size = np.array(pool_size)
    if padding == 'valid':
        output_spatial_shape = np.floor((spatial_shape - pool_size) / strides) + 1
        for i in range(len(output_spatial_shape)):
            if i not in none_dims and output_spatial_shape[i] < 0:
                raise ValueError(f'Computed output size would be negative. Received: `inputs.shape={input_shape}` and `pool_size={pool_size}`.')
    elif padding == 'same':
        output_spatial_shape = np.floor((spatial_shape - 1) / strides) + 1
    else:
        raise ValueError(f"Argument `padding` must be either 'valid' or 'same'. Received: padding={padding}")
    output_spatial_shape = [int(i) for i in output_spatial_shape]
    for i in none_dims:
        output_spatial_shape[i] = None
    output_spatial_shape = tuple(output_spatial_shape)
    if data_format == 'channels_last':
        output_shape = (input_shape_origin[0],) + output_spatial_shape + (input_shape_origin[-1],)
    else:
        output_shape = (input_shape_origin[0], input_shape_origin[1]) + output_spatial_shape
    return output_shape