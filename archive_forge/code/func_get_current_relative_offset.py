import functools
from collections import defaultdict
from typing import Callable, Dict
import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload
@classmethod
def get_current_relative_offset(cls):
    return cls.running_state.relative_offset