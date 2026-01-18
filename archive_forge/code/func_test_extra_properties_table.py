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
def test_extra_properties_table():
    out = regionprops_table(SAMPLE_MULTIPLE, intensity_image=INTENSITY_SAMPLE_MULTIPLE, properties=('label',), extra_properties=(intensity_median, pixelcount, bbox_list))
    assert_array_almost_equal(out['intensity_median'], np.array([2.0, 4.0]))
    assert_array_equal(out['pixelcount'], np.array([10, 2]))
    assert out['bbox_list'].dtype == np.object_
    assert out['bbox_list'][0] == [1] * 10
    assert out['bbox_list'][1] == [1] * 1