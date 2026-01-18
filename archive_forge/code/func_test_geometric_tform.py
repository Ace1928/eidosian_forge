import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_geometric_tform():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        _GeometricTransform()
    for i in range(20):
        H = np.random.rand(3, 3) * 100
        H[2, H[2] == 0] += np.finfo(float).eps
        H /= H[2, 2]
        src = np.array([[(H[2, 1] + 1) / -H[2, 0], 1], [1, (H[2, 0] + 1) / -H[2, 1]], [1, 1]])
        tform = ProjectiveTransform(H)
        dst = tform(src)
        assert np.isfinite(dst).all()