import numpy as np
import pytest
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
from skimage._shared.utils import _supported_float_type
from skimage.data import camera, coins
from skimage.filters import (
@pytest.mark.parametrize('c_slice', [slice(None), slice(0, -5), slice(0, -20)])
def test_ip_shape(self, c_slice):
    x = self.img[:, c_slice]
    assert_equal(self.f(x).shape, x.shape)