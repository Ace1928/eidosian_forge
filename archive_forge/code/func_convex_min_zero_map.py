import dataclasses
import itertools
import sympy
from sympy.logic.boolalg import BooleanAtom, Boolean as SympyBoolean
import operator
import math
import logging
import torch
from typing import Union, Dict, Optional, SupportsFloat
from torch._prims_common import dtype_to_type
from .interp import sympy_interp
@classmethod
def convex_min_zero_map(cls, x, fn):
    """Fn is convex and has a minimum at 0."""
    x = ValueRanges.wrap(x)
    if 0 in x:
        return ValueRanges(0, max(fn(x.lower), fn(x.upper)))
    else:
        return cls.monotone_map(x, fn)