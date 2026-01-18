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
def set_correction(unbiased: Optional[bool]=None, correction: Optional[NumberType]=None) -> float:
    if correction is not None and unbiased is not None:
        raise RuntimeError('cannot specify both correction and unbiased arguments')
    elif correction is None and unbiased is None:
        correction = 1.0
    elif correction is None and unbiased is not None:
        correction = 0.0 if unbiased is False else 1.0
    if not isinstance(correction, (IntLike, FloatLike)):
        raise ValueError('correction argument should be integer or float')
    if correction < 0:
        raise ValueError('correction argument should be non-negative')
    return sym_float(correction)