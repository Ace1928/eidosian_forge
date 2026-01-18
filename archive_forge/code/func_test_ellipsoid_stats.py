import numpy as np
from skimage._shared.testing import assert_array_equal, assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats, rectangle
from skimage._shared import testing
def test_ellipsoid_stats():
    vol, surf = ellipsoid_stats(6, 10, 16)
    assert_allclose(1280 * np.pi, vol, atol=0.0001)
    assert_allclose(1383.28, surf, atol=0.01)
    vol, surf = ellipsoid_stats(16, 6, 10)
    assert_allclose(1280 * np.pi, vol, atol=0.0001)
    assert_allclose(1383.28, surf, atol=0.01)
    vol, surf = ellipsoid_stats(17, 27, 169)
    assert_allclose(103428 * np.pi, vol, atol=0.0001)
    assert_allclose(37426.3, surf, atol=0.1)