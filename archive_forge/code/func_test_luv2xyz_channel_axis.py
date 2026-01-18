import colorsys
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.color import (
from skimage.util import img_as_float, img_as_ubyte, img_as_float32
@pytest.mark.parametrize('channel_axis', [0, 1, -1, -2])
def test_luv2xyz_channel_axis(self, channel_axis):
    luv = np.moveaxis(self.luv_array, source=-1, destination=channel_axis)
    xyz = luv2xyz(luv, channel_axis=channel_axis)
    xyz = np.moveaxis(xyz, source=channel_axis, destination=-1)
    assert_array_almost_equal(xyz, self.xyz_array, decimal=3)