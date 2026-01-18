import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
def test_gray_morphology(self):
    expected = dict(np.load(fetch('data/gray_morph_output.npz')))
    calculated = self._build_expected_output()
    assert_equal(expected, calculated)