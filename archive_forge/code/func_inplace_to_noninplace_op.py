from __future__ import annotations
import operator
import numpy as np
from xarray.core import dtypes, duck_array_ops
def inplace_to_noninplace_op(f):
    return NON_INPLACE_OP[f]