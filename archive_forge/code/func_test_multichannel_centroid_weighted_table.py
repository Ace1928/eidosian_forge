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
def test_multichannel_centroid_weighted_table():
    """Test for https://github.com/scikit-image/scikit-image/issues/6860."""
    intensity_image = INTENSITY_FLOAT_SAMPLE_MULTICHANNEL
    rp0 = regionprops(SAMPLE, intensity_image=intensity_image[..., 0])[0]
    rp1 = regionprops(SAMPLE, intensity_image=intensity_image[..., 0:1])[0]
    rpm = regionprops(SAMPLE, intensity_image=intensity_image)[0]
    np.testing.assert_almost_equal(rp0.centroid_weighted, np.squeeze(rp1.centroid_weighted))
    np.testing.assert_almost_equal(rp0.centroid_weighted, np.array(rpm.centroid_weighted)[:, 0])
    assert np.shape(rp0.centroid_weighted) == (SAMPLE.ndim,)
    assert np.shape(rp1.centroid_weighted) == (SAMPLE.ndim, 1)
    assert np.shape(rpm.centroid_weighted) == (SAMPLE.ndim, intensity_image.shape[-1])
    table = regionprops_table(SAMPLE, intensity_image=intensity_image, properties=('centroid_weighted',))
    assert len(table) == np.size(rpm.centroid_weighted)