import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
@staticmethod
def lookup_schemas(name: str) -> Optional[Tuple[FunctionSchema, ...]]:
    try:
        if '::' not in name:
            return None
        return tuple(torch._C._jit_get_schemas_for_operator(name))
    except RuntimeError:
        return None