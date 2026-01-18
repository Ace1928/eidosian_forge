import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong
def getdtype(dtype, a=None, default=None):
    """Function used to simplify argument processing. If 'dtype' is not
    specified (is None), returns a.dtype; otherwise returns a np.dtype
    object created from the specified dtype argument. If 'dtype' and 'a'
    are both None, construct a data type out of the 'default' parameter.
    Furthermore, 'dtype' must be in 'allowed' set.
    """
    if dtype is None:
        try:
            newdtype = a.dtype
        except AttributeError as e:
            if default is not None:
                newdtype = np.dtype(default)
            else:
                raise TypeError('could not interpret data type') from e
    else:
        newdtype = np.dtype(dtype)
        if newdtype == np.object_:
            raise ValueError('object dtype is not supported by sparse matrices')
    return newdtype