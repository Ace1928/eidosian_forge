import functools
from collections import defaultdict
from typing import Callable, Dict
import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload
@classmethod
def mark_beginning_of_backward(cls):
    cls.running_state = cls.bwd_state