import functools
from collections import defaultdict
from typing import Callable, Dict
import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload
@classmethod
def set_state_from_tensor(cls, x):
    cls.running_state.set_state_from_tensor(x)