import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    assert e + p == total_bits - has_sign
    evalues = []
    pvalues = []
    for i, val in enumerate(range(-2 ** (exponent_bits - has_sign), 2 ** (exponent_bits - has_sign), 1)):
        evalues.append(2 ** val)
    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))
    bias = 2 ** (exponent_bits - 1)
    for evalue in range(2 ** exponent_bits):
        for bit_pattern in lst:
            value = 1 if evalue != 0 else 0
            for i, pval in enumerate(list(bit_pattern)):
                value += pval * 2 ** (-(i + 1))
            if evalue == 0:
                value = value * 2 ** (-bias)
            else:
                value = value * 2 ** (-(evalue - bias - 1))
            values.append(value)
            if signed:
                values.append(-value)
    assert len(values) == 2 ** total_bits
    values.sort()
    if total_bits < 8:
        gap = 256 - len(values)
        for i in range(gap):
            values.append(0)
    values.sort()
    code = torch.Tensor(values)
    code /= code.max()
    return code