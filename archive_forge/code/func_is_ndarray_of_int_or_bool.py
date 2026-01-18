import numbers
from functools import reduce
from operator import mul
import numpy as np
def is_ndarray_of_int_or_bool(obj):
    return isinstance(obj, np.ndarray) and (np.issubdtype(obj.dtype, np.integer) or np.issubdtype(obj.dtype, np.bool_))