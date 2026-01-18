import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong
def asmatrix(data, dtype=None):
    if isinstance(data, np.matrix) and (dtype is None or data.dtype == dtype):
        return data
    return np.asarray(data, dtype=dtype).view(np.matrix)