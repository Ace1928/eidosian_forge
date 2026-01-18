import os
import sys
import warnings
import numpy as np
import pytest
from ..casting import sctypes
from ..testing import (
def test_assert_allclose_safely():
    assert_allclose_safely([1, 1], [1, 1])
    assert_allclose_safely(1, 1)
    assert_allclose_safely(1, [1, 1])
    assert_allclose_safely([1, 1], 1 + 1e-06)
    with pytest.raises(AssertionError):
        assert_allclose_safely([1, 1], 1 + 0.0001)
    a = np.ones((2, 3))
    b = np.ones((3, 2, 3))
    eps = np.finfo(np.float64).eps
    a[0, 0] = 1 + eps
    assert_allclose_safely(a, b)
    a[0, 0] = 1 + 1.1e-05
    with pytest.raises(AssertionError):
        assert_allclose_safely(a, b)
    a[0, 0] = np.nan
    b[:, 0, 0] = np.nan
    assert_allclose_safely(a, b)
    with pytest.raises(AssertionError):
        assert_allclose_safely(a, b, match_nans=False)
    b[0, 0, 0] = 1
    with pytest.raises(AssertionError):
        assert_allclose_safely(a, b)
    for dtt in sctypes['float']:
        a = np.array([-np.inf, 1, np.inf], dtype=dtt)
        b = np.array([-np.inf, 1, np.inf], dtype=dtt)
        assert_allclose_safely(a, b)
        b[1] = 0
        with pytest.raises(AssertionError):
            assert_allclose_safely(a, b)
    assert_allclose_safely([], [])