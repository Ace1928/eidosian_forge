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
def test_iterate_all_props():
    region = regionprops(SAMPLE)[0]
    p0 = {p: region[p] for p in region}
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0]
    p1 = {p: region[p] for p in region}
    assert len(p0) < len(p1)