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
def children_fn(e: _ProfilerEvent):
    if leaf_op(e) or e.tag == _EventType.Allocation:
        leaf_events.append(e)
        return []
    return e.children