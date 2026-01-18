import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_euler_angle_consistency():
    angles = np.random.random((3,)) * 2 * np.pi - np.pi
    euclid = EuclideanTransform(rotation=angles, dimensionality=3)
    similar = SimilarityTransform(rotation=angles, dimensionality=3)
    assert_array_almost_equal(euclid, similar)