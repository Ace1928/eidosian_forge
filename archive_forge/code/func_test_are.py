import numpy as np
import pytest
from skimage.metrics import (
from skimage._shared.testing import (
def test_are():
    im_true = np.array([[2, 1], [1, 2]])
    im_test = np.array([[1, 2], [3, 1]])
    assert_almost_equal(adapted_rand_error(im_true, im_test), (0.3333333, 0.5, 1.0))
    assert_almost_equal(adapted_rand_error(im_true, im_test, alpha=0), (0, 0.5, 1.0))
    assert_almost_equal(adapted_rand_error(im_true, im_test, alpha=1), (0.5, 0.5, 1.0))
    with pytest.raises(ValueError):
        adapted_rand_error(im_true, im_test, alpha=1.01)
    with pytest.raises(ValueError):
        adapted_rand_error(im_true, im_test, alpha=-0.01)