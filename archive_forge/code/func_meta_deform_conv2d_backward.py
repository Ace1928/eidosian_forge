import functools
import torch
import torch._custom_ops
import torch.library
import torchvision.extension  # noqa: F401
@register_meta('_deform_conv2d_backward')
def meta_deform_conv2d_backward(grad, input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, offset_groups, use_mask):
    grad_input = input.new_empty(input.shape)
    grad_weight = weight.new_empty(weight.shape)
    grad_offset = offset.new_empty(offset.shape)
    grad_mask = mask.new_empty(mask.shape)
    grad_bias = bias.new_empty(bias.shape)
    return (grad_input, grad_weight, grad_offset, grad_mask, grad_bias)