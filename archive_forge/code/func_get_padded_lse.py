import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
def get_padded_lse(self, pad_to: int, force_pad_inf: bool=False) -> torch.Tensor:
    pad_amount = (pad_to - self.lse.shape[2] % pad_to) % pad_to
    lse = self.lse
    if pad_amount > 0:
        if force_pad_inf:
            lse = lse[:, :, :self.out.shape[1]]
            pad_amount = (pad_to - lse.shape[2] % pad_to) % pad_to
        lse = torch.nn.functional.pad(lse, [0, pad_amount], value=math.inf)
    elif force_pad_inf and self.out.shape[1] != lse.shape[2]:
        lse[:, :, self.out.shape[1]:].fill_(math.inf)
    return lse