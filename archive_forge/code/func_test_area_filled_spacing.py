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
def test_area_filled_spacing():
    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[7, -3] = 0
    spacing = (2, 1.2)
    area = regionprops(SAMPLE, spacing=spacing)[0].area_filled
    assert area == np.sum(SAMPLE) * np.prod(spacing)
    area = regionprops(SAMPLE_mod, spacing=spacing)[0].area_filled
    assert area == np.sum(SAMPLE) * np.prod(spacing)