from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('dtype', [np.int64, np.float32])
@pytest.mark.parametrize('scalar', [param(np.timedelta64('NaT', 's'), id='timedelta64[s](NaT)'), param(np.timedelta64(123, 's'), id='timedelta64[s]'), param(np.datetime64('NaT', 'generic'), id='datetime64[generic](NaT)'), param(np.datetime64(1, 'D'), id='datetime64[D]')])
def test_coercion_basic(self, dtype, scalar):
    arr = np.array(scalar, dtype=dtype)
    cast = np.array(scalar).astype(dtype)
    assert_array_equal(arr, cast)
    ass = np.ones((), dtype=dtype)
    if issubclass(dtype, np.integer):
        with pytest.raises(TypeError):
            ass[()] = scalar
    else:
        ass[()] = scalar
        assert_array_equal(ass, cast)