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
def test_feret_diameter_max_3d():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[2:-2, 2:-2] = 1
    img_3d = np.dstack((img,) * 3)
    feret_diameter_max = regionprops(img_3d)[0].feret_diameter_max
    assert np.abs(feret_diameter_max - np.sqrt((16 - 1) ** 2 + 16 ** 2 + (3 - 1) ** 2)) < 1e-06
    spacing = (1, 2, 3)
    feret_diameter_max = regionprops(img_3d, spacing=spacing)[0].feret_diameter_max
    assert np.abs(feret_diameter_max - np.sqrt((spacing[0] * (16 - 1)) ** 2 + (spacing[1] * (16 - 0)) ** 2 + (spacing[2] * (3 - 1)) ** 2)) < 1e-06
    assert np.abs(feret_diameter_max - np.sqrt((spacing[0] * (16 - 1)) ** 2 + (spacing[1] * (16 - 1)) ** 2 + (spacing[2] * (3 - 0)) ** 2)) > 1e-06
    assert np.abs(feret_diameter_max - np.sqrt((spacing[0] * (16 - 0)) ** 2 + (spacing[1] * (16 - 1)) ** 2 + (spacing[2] * (3 - 1)) ** 2)) > 1e-06