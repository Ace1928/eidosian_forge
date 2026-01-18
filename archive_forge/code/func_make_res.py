import collections
import numpy as np
from numba.core import types
@wrap
def make_res(A):
    return np.arange(A.size)