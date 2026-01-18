import functools
from collections import defaultdict
from typing import Callable, Dict
import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload
@staticmethod
def multiple_of_4(offset):
    return (offset + 3) // 4 * 4