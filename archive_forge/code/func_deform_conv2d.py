import math
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torchvision.extension import _assert_has_ops
from ..utils import _log_api_usage_once
def deform_conv2d(input: Tensor, offset: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0), dilation: Tuple[int, int]=(1, 1), mask: Optional[Tensor]=None) -> Tensor:
    """
    Performs Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

    Args:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
            offsets to be applied for each position in the convolution kernel.
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]): convolution weights,
            split into groups of size (in_channels // groups)
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
            masks to be applied for each position in the convolution kernel. Default: None

    Returns:
        Tensor[batch_sz, out_channels, out_h, out_w]: result of convolution

    Examples::
        >>> input = torch.rand(4, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kh, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
        >>> mask = torch.rand(4, kh * kw, 8, 8)
        >>> out = deform_conv2d(input, offset, weight, mask=mask)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([4, 5, 8, 8])
    """
    if not torch.jit.is_scripting() and (not torch.jit.is_tracing()):
        _log_api_usage_once(deform_conv2d)
    _assert_has_ops()
    out_channels = weight.shape[0]
    use_mask = mask is not None
    if mask is None:
        mask = torch.zeros((input.shape[0], 1), device=input.device, dtype=input.dtype)
    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, _, _ = input.shape
    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]
    if n_offset_grps == 0:
        raise RuntimeError(f'the shape of the offset tensor at dimension 1 is not valid. It should be a multiple of 2 * weight.size[2] * weight.size[3].\nGot offset.shape[1]={offset.shape[1]}, while 2 * weight.size[2] * weight.size[3]={2 * weights_h * weights_w}')
    return torch.ops.torchvision.deform_conv2d(input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, n_weight_grps, n_offset_grps, use_mask)