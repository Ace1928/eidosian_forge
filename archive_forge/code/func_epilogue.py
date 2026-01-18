import functools
import logging
from typing import cast, List, Tuple
import sympy
import torch
from torch._inductor.select_algorithm import realize_inputs
from torch._inductor.virtualized import V
from ..utils import ceildiv as cdiv, next_power_of_2
def epilogue(acc, bias):
    if alpha != 1:
        acc = V.ops.mul(acc, V.ops.constant(alpha, dtype))
    if beta != 1:
        bias = V.ops.mul(bias, V.ops.constant(beta, dtype))
    return V.ops.add(acc, bias)