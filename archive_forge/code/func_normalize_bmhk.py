import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
def normalize_bmhk(self) -> Tuple[int, ...]:
    if self.query.ndim not in [3, 4, 5]:
        raise ValueError(f'Invalid shape for query: {self.query.shape}. Expected shape [batch, seqlen, head_groups, num_heads_per_group, K], [batch, seqlen, num_heads, K], or [batch, seqlen, K].')
    if self.value.dtype == torch.int32:
        output_shape = tuple(self.query.shape)
    else:
        output_shape = self.query.shape[:-1] + (self.value.shape[-1],)
    if self.query.ndim == 3:
        self.query = self.query.unsqueeze(2)
        self.key = self.key.unsqueeze(2)
        self.value = self.value.unsqueeze(2)
        self.attn_bias = _attn_bias_apply(self.attn_bias, partial(torch.unsqueeze, dim=1))
    return output_shape