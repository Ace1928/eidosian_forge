import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_euler_rotation():
    v = [0, 10, 0]
    angles = np.radians([90, 45, 45])
    expected = [-5, -5, 7.1]
    R = _euler_rotation_matrix(angles)
    assert_almost_equal(R @ v, expected, decimal=1)