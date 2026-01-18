from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('arraylike', arraylikes())
def test_0d_object_special_case(self, arraylike):
    arr = np.array(0.0)
    obj = arraylike(arr)
    res = np.array(obj, dtype=object)
    assert_array_equal(arr, res)
    res = np.array([obj], dtype=object)
    assert res[0] is obj