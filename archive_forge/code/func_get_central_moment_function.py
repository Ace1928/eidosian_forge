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
def get_central_moment_function(img, spacing=(1, 1)):
    rows, cols = img.shape
    Y, X = np.meshgrid(np.linspace(0, rows * spacing[0], rows, endpoint=False), np.linspace(0, cols * spacing[1], cols, endpoint=False), indexing='ij')
    Mpq = get_moment_function(img, spacing=spacing)
    cY = Mpq(1, 0) / Mpq(0, 0)
    cX = Mpq(0, 1) / Mpq(0, 0)
    return lambda p, q: np.sum((Y - cY) ** p * (X - cX) ** q * img)