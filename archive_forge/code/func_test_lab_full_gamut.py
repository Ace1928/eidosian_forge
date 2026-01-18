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
def test_lab_full_gamut(self):
    a, b = np.meshgrid(np.arange(-100, 100), np.arange(-100, 100))
    L = np.ones(a.shape)
    lab = np.dstack((L, a, b))
    regex = 'Conversion from CIE-LAB to XYZ color space resulted in \\d+ negative Z values that have been clipped to zero'
    for value in [0, 10, 20]:
        lab[:, :, 0] = value
        with pytest.warns(UserWarning, match=regex) as record:
            lab2xyz(lab)
        assert_stacklevel(record)