import numpy as np
from scipy._lib import doccer
from . import _byteordercodes as boc
def arr_dtype_number(arr, num):
    """ Return dtype for given number of items per element"""
    return np.dtype(arr.dtype.str[:2] + str(num))