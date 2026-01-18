import torch
from .modules.utils import _single, _pair, _triple
def conv1d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1):
    """Compute the gradient of conv1d with respect to the input of the convolution.

    This is same as the 1D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv1d_input(input.shape, weight, grad_output)

    """
    input = grad_output.new_empty(1).expand(input_size)
    return torch.ops.aten.convolution_backward(grad_output, input, weight, None, _single(stride), _single(padding), _single(dilation), False, [0], groups, (True, False, False))[0]