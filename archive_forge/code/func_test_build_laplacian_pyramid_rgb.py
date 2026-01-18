import math
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.transform import pyramids
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2, -3])
def test_build_laplacian_pyramid_rgb(channel_axis):
    image = data.astronaut()
    rows, cols, dim = image.shape
    image = np.moveaxis(image, source=-1, destination=channel_axis)
    pyramid = pyramids.pyramid_laplacian(image, downscale=2, channel_axis=channel_axis)
    for layer, out in enumerate(pyramid):
        layer_shape = [rows / 2 ** layer, cols / 2 ** layer]
        layer_shape.insert(channel_axis % image.ndim, dim)
        assert out.shape == tuple(layer_shape)