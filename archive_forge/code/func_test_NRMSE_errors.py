import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage.metrics import (
def test_NRMSE_errors():
    x = np.ones(4)
    with pytest.raises(ValueError):
        normalized_root_mse(x[:-1], x)
    with pytest.raises(ValueError):
        normalized_root_mse(x, x, normalization='foo')