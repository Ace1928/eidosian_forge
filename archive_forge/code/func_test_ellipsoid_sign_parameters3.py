import numpy as np
from skimage._shared.testing import assert_array_equal, assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats, rectangle
from skimage._shared import testing
def test_ellipsoid_sign_parameters3():
    with testing.raises(ValueError):
        ellipsoid(-3, -2, 2)