import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
def bmk2bmhk(tensor, num_heads: int) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor
    return tensor.reshape([tensor.shape[0] // num_heads, num_heads, tensor.shape[1], tensor.shape[2]]).permute((0, 2, 1, 3))