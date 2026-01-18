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
def increasing_map(cls, x, fn):
    """Increasing: x <= y => f(x) <= f(y)."""
    x = cls.wrap(x)
    return ValueRanges(fn(x.lower), fn(x.upper))