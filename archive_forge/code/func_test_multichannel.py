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
def test_multichannel():
    """Test that computing multichannel properties works."""
    astro = data.astronaut()[::4, ::4]
    astro_green = astro[..., 1]
    labels = slic(astro.astype(float), start_label=1)
    segment_idx = np.max(labels) // 2
    region = regionprops(labels, astro_green, extra_properties=[intensity_median])[segment_idx]
    region_multi = regionprops(labels, astro, extra_properties=[intensity_median])[segment_idx]
    for prop in list(PROPS.keys()) + ['intensity_median']:
        p = region[prop]
        p_multi = region_multi[prop]
        if np.shape(p) == np.shape(p_multi):
            assert_array_equal(p, p_multi)
        else:
            assert_allclose(p, np.asarray(p_multi)[..., 1], rtol=1e-12, atol=1e-12)