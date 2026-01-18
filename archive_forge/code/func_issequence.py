import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong
def issequence(t) -> bool:
    return isinstance(t, (list, tuple)) and (len(t) == 0 or np.isscalar(t[0])) or (isinstance(t, np.ndarray) and t.ndim == 1)