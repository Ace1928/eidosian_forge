import math
import warnings
from torch import Tensor
import torch
from typing import Optional as _Optional
def eye_(tensor):
    """Fill the 2-dimensional input `Tensor` with the identity matrix.

    Preserves the identity of the inputs in `Linear` layers, where as
    many inputs are preserved as possible.

    Args:
        tensor: a 2-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.eye_(w)
    """
    if tensor.ndimension() != 2:
        raise ValueError('Only tensors with 2 dimensions are supported')
    with torch.no_grad():
        torch.eye(*tensor.shape, out=tensor, requires_grad=tensor.requires_grad)
    return tensor