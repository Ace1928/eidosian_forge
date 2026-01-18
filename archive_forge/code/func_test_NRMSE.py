import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage.metrics import (
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_NRMSE(dtype):
    x = np.ones(4, dtype=dtype)
    y = np.asarray([0.0, 2.0, 2.0, 2.0], dtype=dtype)
    nrmse = normalized_root_mse(y, x, normalization='mean')
    assert nrmse.dtype == np.float64
    assert_equal(nrmse, 1 / np.mean(y, dtype=np.float64))
    assert_equal(normalized_root_mse(y, x, normalization='euclidean'), 1 / np.sqrt(3))
    assert_equal(normalized_root_mse(y, x, normalization='min-max'), 1 / (y.max() - y.min()))
    assert_almost_equal(normalized_root_mse(y, np.float32(x), normalization='min-max'), 1 / (y.max() - y.min()))