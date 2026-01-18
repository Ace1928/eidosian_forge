import numpy as np
from skimage._shared.testing import assert_array_equal, assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats, rectangle
from skimage._shared import testing
def test_ellipsoid_levelset():
    test = ellipsoid(2, 2, 2, levelset=True)[1:-1, 1:-1, 1:-1]
    test_anisotropic = ellipsoid(2, 2, 4, spacing=(1.0, 1.0, 2.0), levelset=True)
    test_anisotropic = test_anisotropic[1:-1, 1:-1, 1:-1]
    expected = np.array([[[2.0, 1.25, 1.0, 1.25, 2.0], [1.25, 0.5, 0.25, 0.5, 1.25], [1.0, 0.25, 0.0, 0.25, 1.0], [1.25, 0.5, 0.25, 0.5, 1.25], [2.0, 1.25, 1.0, 1.25, 2.0]], [[1.25, 0.5, 0.25, 0.5, 1.25], [0.5, -0.25, -0.5, -0.25, 0.5], [0.25, -0.5, -0.75, -0.5, 0.25], [0.5, -0.25, -0.5, -0.25, 0.5], [1.25, 0.5, 0.25, 0.5, 1.25]], [[1.0, 0.25, 0.0, 0.25, 1.0], [0.25, -0.5, -0.75, -0.5, 0.25], [0.0, -0.75, -1.0, -0.75, 0.0], [0.25, -0.5, -0.75, -0.5, 0.25], [1.0, 0.25, 0.0, 0.25, 1.0]], [[1.25, 0.5, 0.25, 0.5, 1.25], [0.5, -0.25, -0.5, -0.25, 0.5], [0.25, -0.5, -0.75, -0.5, 0.25], [0.5, -0.25, -0.5, -0.25, 0.5], [1.25, 0.5, 0.25, 0.5, 1.25]], [[2.0, 1.25, 1.0, 1.25, 2.0], [1.25, 0.5, 0.25, 0.5, 1.25], [1.0, 0.25, 0.0, 0.25, 1.0], [1.25, 0.5, 0.25, 0.5, 1.25], [2.0, 1.25, 1.0, 1.25, 2.0]]])
    assert_allclose(test, expected)
    assert_allclose(test_anisotropic, expected)