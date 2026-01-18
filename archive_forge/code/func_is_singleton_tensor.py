import copy
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Tuple, cast
import torch
from fairscale.nn.misc import FlattenParamsWrapper
def is_singleton_tensor(x: Any) -> bool:
    """Is x a dimensionless tensor?"""
    return torch.is_tensor(x) and x.dim() == 0