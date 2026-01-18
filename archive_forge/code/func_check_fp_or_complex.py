from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def check_fp_or_complex(dtype: torch.dtype, fn_name: str, allow_low_precision_dtypes: bool=True):
    """
    Checks whether the input is floating point or complex.
    If allow_low_precision_dtypes is True, it allows having float16, bfloat16, and complex32
    """
    torch._check(is_float_dtype(dtype) or is_complex_dtype(dtype), lambda: f'{fn_name}: Expected a floating point or complex tensor as input. Got {dtype}')
    torch._check(allow_low_precision_dtypes or not is_low_precision_dtype(dtype), lambda: f'{fn_name}: Half precision dtypes not supported. Got {dtype}')