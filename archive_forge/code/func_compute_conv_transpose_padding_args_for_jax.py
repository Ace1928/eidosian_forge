import warnings
def compute_conv_transpose_padding_args_for_jax(input_shape, kernel_shape, strides, padding, output_padding, dilation_rate):
    num_spatial_dims = len(input_shape) - 2
    kernel_spatial_shape = kernel_shape[:-2]
    jax_padding = []
    for i in range(num_spatial_dims):
        output_padding_i = output_padding if output_padding is None or isinstance(output_padding, int) else output_padding[i]
        strides_i = strides if isinstance(strides, int) else strides[i]
        dilation_rate_i = dilation_rate if isinstance(dilation_rate, int) else dilation_rate[i]
        pad_left, pad_right = _convert_conv_tranpose_padding_args_from_keras_to_jax(kernel_size=kernel_spatial_shape[i], stride=strides_i, dilation_rate=dilation_rate_i, padding=padding, output_padding=output_padding_i)
        jax_padding.append((pad_left, pad_right))
    return jax_padding