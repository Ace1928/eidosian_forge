import math
import sympy
import torch
from torch.utils._sympy.value_ranges import ValueRanges
from .ir import LoopBody
from .utils import dominated_nodes
def skip_filter(node):
    return node.target == 'to_dtype' and node.args[2] in (torch.int32, torch.float32, torch.float64)