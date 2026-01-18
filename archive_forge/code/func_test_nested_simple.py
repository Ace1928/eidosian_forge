from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
def test_nested_simple(self):
    initial = [1.2]
    nested = initial
    for i in range(np.MAXDIMS - 1):
        nested = [nested]
    arr = np.array(nested, dtype='float64')
    assert arr.shape == (1,) * np.MAXDIMS
    with pytest.raises(ValueError):
        np.array([nested], dtype='float64')
    with pytest.raises(ValueError, match='.*would exceed the maximum'):
        np.array([nested])
    arr = np.array([nested], dtype=object)
    assert arr.dtype == np.dtype('O')
    assert arr.shape == (1,) * np.MAXDIMS
    assert arr.item() is initial