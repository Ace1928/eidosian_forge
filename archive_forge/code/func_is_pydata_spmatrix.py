import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong
def is_pydata_spmatrix(m) -> bool:
    """
    Check whether object is pydata/sparse matrix, avoiding importing the module.
    """
    base_cls = getattr(sys.modules.get('sparse'), 'SparseArray', None)
    return base_cls is not None and isinstance(m, base_cls)