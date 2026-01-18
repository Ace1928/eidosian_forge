import math
import warnings
from torch import Tensor
import torch
from typing import Optional as _Optional
def sparse_(tensor, sparsity, std=0.01, generator: _Optional[torch.Generator]=None):
    """Fill the 2D input `Tensor` as a sparse matrix.

    The non-zero elements will be drawn from the normal distribution
    :math:`\\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError('Only tensors with 2 dimensions are supported')
    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))
    with torch.no_grad():
        tensor.normal_(0, std, generator=generator)
        for col_idx in range(cols):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor