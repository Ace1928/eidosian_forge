import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('dtype', 'edfgFDG')
def test_huge_float(dtype):
    field = '0' * 1000 + '.123456789'
    dtype = np.dtype(dtype)
    value = np.loadtxt([field], dtype=dtype)[()]
    assert value == dtype.type('0.123456789')