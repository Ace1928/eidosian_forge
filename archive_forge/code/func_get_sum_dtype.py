import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong
def get_sum_dtype(dtype: np.dtype) -> np.dtype:
    """Mimic numpy's casting for np.sum"""
    if dtype.kind == 'u' and np.can_cast(dtype, np.uint):
        return np.uint
    if np.can_cast(dtype, np.int_):
        return np.int_
    return dtype