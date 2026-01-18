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
@classmethod
def from_allocation(cls, alloc: _ExtraFields_Allocation) -> Optional['TensorKey']:
    return cls._make(alloc.id, alloc.ptr, alloc.allocation_id, alloc.device)