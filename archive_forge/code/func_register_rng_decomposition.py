import functools
from collections import defaultdict
from typing import Callable, Dict
import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload
def register_rng_decomposition(aten_op):
    return decomp.register_decomposition(aten_op, rng_decompositions)