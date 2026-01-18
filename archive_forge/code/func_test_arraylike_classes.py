from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
def test_arraylike_classes(self):
    arr = np.array(np.int64)
    assert arr[()] is np.int64
    arr = np.array([np.int64])
    assert arr[0] is np.int64

    class ArrayLike:

        @property
        def __array_interface__(self):
            pass

        @property
        def __array_struct__(self):
            pass

        def __array__(self):
            pass
    arr = np.array(ArrayLike)
    assert arr[()] is ArrayLike
    arr = np.array([ArrayLike])
    assert arr[0] is ArrayLike