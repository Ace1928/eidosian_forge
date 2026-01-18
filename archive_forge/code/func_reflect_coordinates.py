import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
def reflect_coordinates(coords: Tensor, twice_low: int, twice_high: int) -> Tensor:
    if twice_low == twice_high:
        return torch.zeros_like(coords)
    coords_min = twice_low / 2
    coords_span = (twice_high - twice_low) / 2
    coords2 = (coords - coords_min).abs()
    extra = torch.fmod(coords2, coords_span)
    flips = (coords2 / coords_span).floor().to(dtype=torch.int8)
    return torch.where(flips & 1 == 0, extra + coords_min, coords_span + coords_min - extra)