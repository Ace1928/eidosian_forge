import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.interpolate import (griddata, NearestNDInterpolator,
def test_1d_unsorted(self):
    x = np.array([2.5, 1, 4.5, 5, 6, 3])
    y = np.array([1, 2, 0, 3.9, 2, 1])
    for method in ('nearest', 'linear', 'cubic'):
        assert_allclose(griddata(x, y, x, method=method), y, err_msg=method, atol=1e-10)
        assert_allclose(griddata(x.reshape(6, 1), y, x, method=method), y, err_msg=method, atol=1e-10)
        assert_allclose(griddata((x,), y, (x,), method=method), y, err_msg=method, atol=1e-10)