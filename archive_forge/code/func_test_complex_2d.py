import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.interpolate import (griddata, NearestNDInterpolator,
def test_complex_2d(self):
    x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
    y = np.arange(x.shape[0], dtype=np.float64)
    y = y - 2j * y[::-1]
    xi = x[:, None, :] + np.array([0, 0, 0])[None, :, None]
    for method in ('nearest', 'linear', 'cubic'):
        for rescale in (True, False):
            msg = repr((method, rescale))
            yi = griddata(x, y, xi, method=method, rescale=rescale)
            assert_equal(yi.shape, (5, 3), err_msg=msg)
            assert_allclose(yi, np.tile(y[:, None], (1, 3)), atol=1e-14, err_msg=msg)