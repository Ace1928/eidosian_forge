import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage.metrics import (
def test_nmi_random_3d():
    random1, random2 = np.random.random((2, 10, 100, 100))
    assert_almost_equal(normalized_mutual_information(random1, random2, bins=10), 1, decimal=2)