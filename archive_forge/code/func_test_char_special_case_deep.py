from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
def test_char_special_case_deep(self):
    nested = ['string']
    for i in range(np.MAXDIMS - 2):
        nested = [nested]
    arr = np.array(nested, dtype='c')
    assert arr.shape == (1,) * (np.MAXDIMS - 1) + (6,)
    with pytest.raises(ValueError):
        np.array([nested], dtype='c')