import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong
def downcast_intp_index(arr):
    """
    Down-cast index array to np.intp dtype if it is of a larger dtype.

    Raise an error if the array contains a value that is too large for
    intp.
    """
    if arr.dtype.itemsize > np.dtype(np.intp).itemsize:
        if arr.size == 0:
            return arr.astype(np.intp)
        maxval = arr.max()
        minval = arr.min()
        if maxval > np.iinfo(np.intp).max or minval < np.iinfo(np.intp).min:
            raise ValueError('Cannot deal with arrays with indices larger than the machine maximum address size (e.g. 64-bit indices on 32-bit machine).')
        return arr.astype(np.intp)
    return arr