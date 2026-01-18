import functools
import operator
from typing import cast, Iterable, List, Sequence, Tuple, Union
import torch
from torch.distributed._tensor._collective_utils import redistribute_cost
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.op_schema import OpStrategy
from torch.distributed._tensor.placement_types import (
def normalize_to_torch_size(size) -> torch.Size:
    """
    Unify variable types of size argument to torch.Size
    Acceptable types include:
        int, Sequence[int], Tuple[int], Tuple[Sequence[int]],
        or torch.Size
    """
    if isinstance(size, torch.Size):
        return size
    if isinstance(size, int):
        torch_size = [size]
    elif len(size) == 1 and isinstance(size[0], Sequence):
        torch_size = list(size[0])
    else:
        torch_size = list(size)
    return torch.Size(torch_size)