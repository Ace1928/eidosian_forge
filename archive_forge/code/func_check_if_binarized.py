import functools
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, conv3d, pad, unfold
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE
def check_if_binarized(x: Tensor) -> None:
    """Check if the input is binarized.

    Example:
        >>> from torchmetrics.functional.segmentation.utils import check_if_binarized
        >>> import torch
        >>> check_if_binarized(torch.tensor([0, 1, 1, 0]))

    """
    if not torch.all(x.bool() == x):
        raise ValueError('Input x should be binarized')