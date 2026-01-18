import functools
import logging
from typing import cast, List, Tuple
import sympy
import torch
from torch._inductor.select_algorithm import realize_inputs
from torch._inductor.virtualized import V
from ..utils import ceildiv as cdiv, next_power_of_2
def mm_grid(m, n, meta):
    """
    The CUDA grid size for matmul triton templates.
    """
    return (cdiv(m, meta['BLOCK_M']) * cdiv(n, meta['BLOCK_N']), 1, 1)