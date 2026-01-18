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
def coordinatewise_monotone_map(cls, x, y, fn):
    """It's increasing or decreasing on each coordinate."""
    x, y = (cls.wrap(x), cls.wrap(y))
    products = [fn(a, b) for a, b in itertools.product([x.lower, x.upper], [y.lower, y.upper])]
    return ValueRanges(min(products), max(products))