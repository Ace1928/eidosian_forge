import math
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.transform import pyramids
def test_build_gaussian_pyramid_gray_defaults():
    rows, cols = image_gray.shape
    pyramid = pyramids.pyramid_gaussian(image_gray)
    for layer, out in enumerate(pyramid):
        layer_shape = (rows / 2 ** layer, cols / 2 ** layer)
        assert_array_equal(out.shape, layer_shape)