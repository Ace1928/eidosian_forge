import pytest
import operator
import numpy as np
from numpy.testing import assert_array_equal
@pytest.mark.parametrize('str_dt', ['S', 'U'])
@pytest.mark.parametrize('float_dt', np.typecodes['AllFloat'])
def test_float_to_string_cast(str_dt, float_dt):
    float_dt = np.dtype(float_dt)
    fi = np.finfo(float_dt)
    arr = np.array([np.nan, np.inf, -np.inf, fi.max, fi.min], dtype=float_dt)
    expected = ['nan', 'inf', '-inf', repr(fi.max), repr(fi.min)]
    if float_dt.kind == 'c':
        expected = [f'({r}+0j)' for r in expected]
    res = arr.astype(str_dt)
    assert_array_equal(res, np.array(expected, dtype=str_dt))