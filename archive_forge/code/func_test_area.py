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
@pytest.mark.parametrize('sample,spacing', [(SAMPLE, None), (SAMPLE, 1), (SAMPLE, (1, 1)), (SAMPLE, (1, 2)), (SAMPLE_3D, None), (SAMPLE_3D, 1), (SAMPLE_3D, (2, 1, 3))])
def test_area(sample, spacing):
    area = regionprops(sample, spacing=spacing)[0].area
    desired = np.sum(sample * (np.prod(spacing) if spacing else 1))
    assert area == desired