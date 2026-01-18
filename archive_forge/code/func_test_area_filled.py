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
def test_area_filled():
    area = regionprops(SAMPLE)[0].area_filled
    assert area == np.sum(SAMPLE)