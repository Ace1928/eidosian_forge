import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong
def getdata(obj, dtype=None, copy=False) -> np.ndarray:
    """
    This is a wrapper of `np.array(obj, dtype=dtype, copy=copy)`
    that will generate a warning if the result is an object array.
    """
    data = np.array(obj, dtype=dtype, copy=copy)
    getdtype(data.dtype)
    return data