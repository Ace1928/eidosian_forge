import functools
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, conv3d, pad, unfold
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE
def table_contour_length(spacing: Tuple[int, int], device: Optional[torch.device]=None) -> Tuple[Tensor, Tensor]:
    """Create a table that maps neighbour codes to the contour length of the corresponding contour.

    Adopted from:
    https://github.com/deepmind/surface-distance/blob/master/surface_distance/lookup_tables.py

    Args:
        spacing: The spacing between pixels along each spatial dimension. Should be a tuple of length 2.
        device: The device on which the table should be created.

    Returns:
        A tuple containing as its first element the table that maps neighbour codes to the contour length of the
        corresponding contour and as its second element the kernel used to compute the neighbour codes.

    Example::
        >>> from torchmetrics.functional.segmentation.utils import table_contour_length
        >>> table, kernel = table_contour_length((2,2))
        >>> table
        tensor([0.0000, 1.4142, 1.4142, 2.0000, 1.4142, 2.0000, 2.8284, 1.4142, 1.4142,
                2.8284, 2.0000, 1.4142, 2.0000, 1.4142, 1.4142, 0.0000])
        >>> kernel
        tensor([[[[8, 4],
                  [2, 1]]]])

    """
    if not isinstance(spacing, tuple) and len(spacing) != 2:
        raise ValueError('The spacing must be a tuple of length 2.')
    first, second = spacing
    diag = 0.5 * math.sqrt(first ** 2 + second ** 2)
    table = torch.zeros(16, dtype=torch.float32, device=device)
    for i in [1, 2, 4, 7, 8, 11, 13, 14]:
        table[i] = diag
    for i in [3, 12]:
        table[i] = second
    for i in [5, 10]:
        table[i] = first
    for i in [6, 9]:
        table[i] = 2 * diag
    kernel = torch.as_tensor([[[[8, 4], [2, 1]]]], device=device)
    return (table, kernel)