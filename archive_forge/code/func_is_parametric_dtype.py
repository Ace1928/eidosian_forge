from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
def is_parametric_dtype(dtype):
    """Returns True if the dtype is a parametric legacy dtype (itemsize
    is 0, or a datetime without units)
    """
    if dtype.itemsize == 0:
        return True
    if issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        if dtype.name.endswith('64'):
            return True
    return False