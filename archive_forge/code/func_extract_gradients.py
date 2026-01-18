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
def extract_gradients(node: _ProfilerEvent) -> Iterator[Tuple[Optional[TensorKey], TensorKey]]:
    for p, p_grad in _extract_parameters_and_gradients(node):
        if p_grad is not None:
            yield (p, p_grad)