from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
def test_0d_generic_special_case(self):

    class ArraySubclass(np.ndarray):

        def __float__(self):
            raise TypeError('e.g. quantities raise on this')
    arr = np.array(0.0)
    obj = arr.view(ArraySubclass)
    res = np.array(obj)
    assert_array_equal(arr, res)
    with pytest.raises(TypeError):
        np.array([obj])
    obj = memoryview(arr)
    res = np.array(obj)
    assert_array_equal(arr, res)
    with pytest.raises(ValueError):
        np.array([obj])