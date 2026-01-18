import numpy as np
import pytest
from numpy import sqrt, ceil
from numpy.testing import assert_almost_equal
from skimage import data
from skimage import img_as_float
from skimage.feature import daisy
def test_daisy_visualization():
    img = img_as_float(data.astronaut()[:32, :32].mean(axis=2))
    descs, descs_img = daisy(img, visualize=True)
    assert descs_img.shape == (32, 32, 3)