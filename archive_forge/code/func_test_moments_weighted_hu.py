import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
def test_moments_weighted_hu():
    whu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].moments_weighted_hu
    ref = np.array([0.31750587329, 0.021417517159, 0.023609322038, 0.001256568336, 8.3014209421e-07, -3.5073773473e-05, -6.7936409056e-06])
    assert_array_almost_equal(whu, ref)
    with testing.raises(NotImplementedError):
        regionprops(SAMPLE, spacing=(2, 1))[0].moments_weighted_hu