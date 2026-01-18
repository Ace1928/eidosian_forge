import warnings
def compute_conv_transpose_output_shape(input_shape, kernel_size, filters, strides, padding, output_padding=None, data_format='channels_last', dilation_rate=1):
    num_spatial_dims = len(input_shape) - 2
    kernel_spatial_shape = kernel_size
    if isinstance(output_padding, int):
        output_padding = (output_padding,) * len(kernel_spatial_shape)
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims
    if data_format == 'channels_last':
        input_spatial_shape = input_shape[1:-1]
    else:
        input_spatial_shape = input_shape[2:]
    output_shape = []
    for i in range(num_spatial_dims):
        current_output_padding = None if output_padding is None else output_padding[i]
        shape_i = _get_output_shape_given_tf_padding(input_size=input_spatial_shape[i], kernel_size=kernel_spatial_shape[i], strides=strides[i], padding=padding, output_padding=current_output_padding, dilation_rate=dilation_rate[i])
        output_shape.append(shape_i)
    if data_format == 'channels_last':
        output_shape = [input_shape[0]] + output_shape + [filters]
    else:
        output_shape = [input_shape[0], filters] + output_shape
    return output_shape