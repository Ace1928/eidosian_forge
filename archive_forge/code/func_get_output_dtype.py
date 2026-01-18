import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
def get_output_dtype(self) -> torch.dtype:
    if self.output_dtype is None:
        if self.is_partial and self.query.dtype is not torch.float64:
            return torch.float32
        return self.query.dtype
    return self.output_dtype