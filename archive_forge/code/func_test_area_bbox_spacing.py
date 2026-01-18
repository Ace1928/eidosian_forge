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
def test_area_bbox_spacing():
    spacing = (0.5, 3)
    padded = np.pad(SAMPLE, 5, mode='constant')
    bbox_area = regionprops(padded, spacing=spacing)[0].area_bbox
    assert_array_almost_equal(bbox_area, SAMPLE.size * np.prod(spacing))