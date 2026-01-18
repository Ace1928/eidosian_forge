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
def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """
    data = []
    non_sign_bits = total_bits - (1 if signed else 1)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1 if signed else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1)
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += (10 ** (-(max_exponent_bits - 1) + i) * means).tolist()
        if signed:
            data += (-10 ** (-(max_exponent_bits - 1) + i) * means).tolist()
    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += (10 ** (-(max_exponent_bits - 1) + i) * means).tolist()
        if signed:
            data += (-10 ** (-(max_exponent_bits - 1) + i) * means).tolist()
    data.append(0)
    data.append(1.0)
    assert len(data) == 2 ** total_bits
    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)
    data.sort()
    return Tensor(data)