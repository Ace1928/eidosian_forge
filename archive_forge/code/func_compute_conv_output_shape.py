import math
import numpy as np
import tree
from keras.src.api_export import keras_export
def compute_conv_output_shape(input_shape, filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1):
    """Compute the output shape of conv ops."""
    if data_format == 'channels_last':
        spatial_shape = input_shape[1:-1]
        kernel_shape = kernel_size + (input_shape[-1], filters)
    else:
        spatial_shape = input_shape[2:]
        kernel_shape = kernel_size + (input_shape[1], filters)
    if len(kernel_shape) != len(input_shape):
        raise ValueError(f'Kernel shape must have the same length as input, but received kernel of shape {kernel_shape} and input of shape {input_shape}.')
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * len(spatial_shape)
    if isinstance(strides, int):
        strides = (strides,) * len(spatial_shape)
    if len(dilation_rate) != len(spatial_shape):
        raise ValueError(f"Dilation must be None, scalar or tuple/list of length of inputs' spatial shape, but received `dilation_rate={dilation_rate}` and input of shape {input_shape}.")
    none_dims = []
    spatial_shape = np.array(spatial_shape)
    for i in range(len(spatial_shape)):
        if spatial_shape[i] is None:
            spatial_shape[i] = -1
            none_dims.append(i)
    kernel_spatial_shape = np.array(kernel_shape[:-2])
    dilation_rate = np.array(dilation_rate)
    if padding == 'valid':
        output_spatial_shape = np.floor((spatial_shape - dilation_rate * (kernel_spatial_shape - 1) - 1) / strides) + 1
        for i in range(len(output_spatial_shape)):
            if i not in none_dims and output_spatial_shape[i] < 0:
                raise ValueError(f'Computed output size would be negative. Received `inputs shape={input_shape}`, `kernel shape={kernel_shape}`, `dilation_rate={dilation_rate}`.')
    elif padding == 'same' or padding == 'causal':
        output_spatial_shape = np.floor((spatial_shape - 1) / strides) + 1
    else:
        raise ValueError(f"`padding` must be either `'valid'` or `'same'`. Received {padding}.")
    output_spatial_shape = [int(i) for i in output_spatial_shape]
    for i in none_dims:
        output_spatial_shape[i] = None
    output_spatial_shape = tuple(output_spatial_shape)
    if data_format == 'channels_last':
        output_shape = (input_shape[0],) + output_spatial_shape + (kernel_shape[-1],)
    else:
        output_shape = (input_shape[0], kernel_shape[-1]) + output_spatial_shape
    return output_shape