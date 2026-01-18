from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
def test_char_special_case(self):
    arr = np.array('string', dtype='c')
    assert arr.shape == (6,)
    assert arr.dtype.char == 'c'
    arr = np.array(['string'], dtype='c')
    assert arr.shape == (1, 6)
    assert arr.dtype.char == 'c'