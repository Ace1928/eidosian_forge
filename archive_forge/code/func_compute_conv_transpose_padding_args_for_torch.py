import warnings
def compute_conv_transpose_padding_args_for_torch(input_shape, kernel_shape, strides, padding, output_padding, dilation_rate):
    num_spatial_dims = len(input_shape) - 2
    kernel_spatial_shape = kernel_shape[:-2]
    torch_paddings = []
    torch_output_paddings = []
    for i in range(num_spatial_dims):
        output_padding_i = output_padding if output_padding is None or isinstance(output_padding, int) else output_padding[i]
        strides_i = strides if isinstance(strides, int) else strides[i]
        dilation_rate_i = dilation_rate if isinstance(dilation_rate, int) else dilation_rate[i]
        torch_padding, torch_output_padding = _convert_conv_tranpose_padding_args_from_keras_to_torch(kernel_size=kernel_spatial_shape[i], stride=strides_i, dilation_rate=dilation_rate_i, padding=padding, output_padding=output_padding_i)
        torch_paddings.append(torch_padding)
        torch_output_paddings.append(torch_output_padding)
    return (torch_paddings, torch_output_paddings)