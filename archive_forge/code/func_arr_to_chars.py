import numpy as np
from scipy._lib import doccer
from . import _byteordercodes as boc
def arr_to_chars(arr):
    """ Convert string array to char array """
    dims = list(arr.shape)
    if not dims:
        dims = [1]
    dims.append(int(arr.dtype.str[2:]))
    arr = np.ndarray(shape=dims, dtype=arr_dtype_number(arr, 1), buffer=arr)
    empties = [arr == np.array('', dtype=arr.dtype)]
    if not np.any(empties):
        return arr
    arr = arr.copy()
    arr[tuple(empties)] = ' '
    return arr