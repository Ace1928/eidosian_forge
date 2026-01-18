import numpy as np
from collections import namedtuple
@wrap(no_cpython_wrapper=True)
def mergesort(arr):
    """Inplace"""
    ws = np.empty(arr.size // 2, dtype=arr.dtype)
    argmergesort_inner(arr, None, ws)
    return arr