import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def deconv(attrs, inputs, proto_obj):
    """Computes transposed convolution of the input tensor."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'kernel_shape': 'kernel', 'strides': 'stride', 'pads': 'pad', 'dilations': 'dilate', 'group': 'num_group'})
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'num_group': 1})
    new_attrs = translation_utils._fix_bias('Deconvolution', new_attrs, len(inputs))
    new_attrs = translation_utils._fix_channels('Deconvolution', new_attrs, inputs, proto_obj)
    kernel = new_attrs['kernel'] if 'kernel' in new_attrs else []
    stride = new_attrs['stride'] if 'stride' in new_attrs else []
    padding = new_attrs['pad'] if 'pad' in new_attrs else []
    dilations = new_attrs['dilate'] if 'dilate' in new_attrs else []
    num_filter = new_attrs['num_filter']
    num_group = new_attrs['num_group']
    no_bias = new_attrs['no_bias'] if 'no_bias' in new_attrs else False
    bias = None if no_bias is True else inputs[2]
    pad_width = (0, 0, 0, 0) + translation_utils._pad_sequence_fix(padding, kernel_dim=len(kernel))
    pad_op = symbol.pad(inputs[0], mode='constant', pad_width=pad_width)
    deconv_op = symbol.Deconvolution(pad_op, inputs[1], bias, kernel=kernel, stride=stride, dilate=dilations, num_filter=num_filter, num_group=num_group, no_bias=no_bias)
    return (deconv_op, new_attrs, inputs)