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
def test_all_props_3d():
    region = regionprops(SAMPLE_3D, INTENSITY_SAMPLE_3D)[0]
    for prop in PROPS:
        try:
            assert_almost_equal(region[prop], getattr(region, PROPS[prop]))
            if prop.lower() == prop:
                assert_almost_equal(getattr(region, prop), getattr(region, PROPS[prop]))
        except (NotImplementedError, TypeError):
            pass