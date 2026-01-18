import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
def sympy_is_channels_last_strides_generic(sizes, strides, dim_order):
    import sympy
    dim = len(sizes)
    if dim != len(dim_order):
        return sympy.false
    m = sympy.Integer(0)
    r = sympy.true
    r &= sympy.Ne(strides[1], 0)
    for d in dim_order:
        r &= sympy.Ne(sizes[d], 0) & (strides[d] >= m)
        if d == 0:
            r &= sympy.Ne(m, strides[1])
        m = strides[d] * sympy.Max(sizes[d], 1)
    return r