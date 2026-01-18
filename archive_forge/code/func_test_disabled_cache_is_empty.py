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
def test_disabled_cache_is_empty():
    SAMPLE_mod = SAMPLE.copy()
    region = regionprops(SAMPLE_mod, cache=False)[0]
    _ = region.image_filled
    assert region._cache == dict()