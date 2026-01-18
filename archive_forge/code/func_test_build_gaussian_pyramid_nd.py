import math
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.transform import pyramids
def test_build_gaussian_pyramid_nd():
    for ndim in [1, 2, 3, 4]:
        img = np.random.randn(*(8,) * ndim)
        original_shape = np.asarray(img.shape)
        pyramid = pyramids.pyramid_gaussian(img, downscale=2, channel_axis=None)
        for layer, out in enumerate(pyramid):
            layer_shape = original_shape / 2 ** layer
            assert_array_equal(out.shape, layer_shape)