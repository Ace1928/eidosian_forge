import math
import warnings
from torch import Tensor
import torch
from typing import Optional as _Optional
def dirac_(tensor, groups=1):
    """Fill the {3, 4, 5}-dimensional input `Tensor` with the Dirac delta function.

    Preserves the identity of the inputs in `Convolutional`
    layers, where as many input channels are preserved as possible. In case
    of groups>1, each group of channels preserves identity

    Args:
        tensor: a {3, 4, 5}-dimensional `torch.Tensor`
        groups (int, optional): number of groups in the conv layer (default: 1)
    Examples:
        >>> w = torch.empty(3, 16, 5, 5)
        >>> nn.init.dirac_(w)
        >>> w = torch.empty(3, 24, 5, 5)
        >>> nn.init.dirac_(w, 3)
    """
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError('Only tensors with 3, 4, or 5 dimensions are supported')
    sizes = tensor.size()
    if sizes[0] % groups != 0:
        raise ValueError('dim 0 must be divisible by groups')
    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])
    with torch.no_grad():
        tensor.zero_()
        for g in range(groups):
            for d in range(min_dim):
                if dimensions == 3:
                    tensor[g * out_chans_per_grp + d, d, tensor.size(2) // 2] = 1
                elif dimensions == 4:
                    tensor[g * out_chans_per_grp + d, d, tensor.size(2) // 2, tensor.size(3) // 2] = 1
                else:
                    tensor[g * out_chans_per_grp + d, d, tensor.size(2) // 2, tensor.size(3) // 2, tensor.size(4) // 2] = 1
    return tensor