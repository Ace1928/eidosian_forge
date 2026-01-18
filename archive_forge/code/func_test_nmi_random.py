import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage.metrics import (
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_nmi_random(dtype):
    rng = np.random.default_rng()
    random1 = rng.random((100, 100)).astype(dtype)
    random2 = rng.random((100, 100)).astype(dtype)
    nmi = normalized_mutual_information(random1, random2, bins=10)
    assert nmi.dtype == np.float64
    assert_almost_equal(nmi, 1, decimal=2)