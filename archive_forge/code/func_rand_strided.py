import contextlib
import dis
import functools
import logging
import os.path
import random
import re
import sys
import types
import unittest
from typing import List, Optional, Sequence, Union
from unittest.mock import patch
import torch
from torch import fx
from torch._dynamo.output_graph import OutputGraph
from . import config, eval_frame, optimize_assert, reset
from .bytecode_transformation import (
from .guards import CheckFunctionManager, GuardedCode
from .utils import same
def rand_strided(size: Sequence[int], stride: Sequence[int], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu', extra_size: int=0):
    needed_size = sum(((shape - 1) * stride for shape, stride in zip(size, stride))) + 1 + extra_size
    if dtype.is_floating_point:
        buffer = torch.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)
    return torch.as_strided(buffer, size, stride)