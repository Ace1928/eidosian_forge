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
def test_intensity_min():
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].intensity_min
    assert_almost_equal(intensity, 1)