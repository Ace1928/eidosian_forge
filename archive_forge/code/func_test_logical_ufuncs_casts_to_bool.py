import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
@pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or, np.logical_xor])
def test_logical_ufuncs_casts_to_bool(self, ufunc):
    a = self._get_array(2.0)
    a[0] = 0.0
    float_equiv = a.astype(float)
    expected = ufunc(float_equiv, float_equiv)
    res = ufunc(a, a)
    assert_array_equal(res, expected)
    expected = ufunc.reduce(float_equiv)
    res = ufunc.reduce(a)
    assert_array_equal(res, expected)
    with pytest.raises(TypeError):
        ufunc(a, a, out=np.empty(a.shape, dtype=int), casting='equiv')