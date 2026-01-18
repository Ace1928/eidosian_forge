import functools
import operator
from typing import cast, Iterable, List, Sequence, Tuple, Union
import torch
from torch.distributed._tensor._collective_utils import redistribute_cost
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.op_schema import OpStrategy
from torch.distributed._tensor.placement_types import (
def normalize_dims(dims: Union[int, Sequence[int]], ndim: int) -> Sequence[int]:
    """Normalize a dim or a sequence of dims, so that they are all positive."""
    if isinstance(dims, int):
        dims = (normalize_dim(dims, ndim),)
    elif isinstance(dims, list):
        dims = [normalize_dim(dim, ndim) for dim in dims]
    elif isinstance(dims, tuple):
        dims = tuple([normalize_dim(dim, ndim) for dim in dims])
    return dims