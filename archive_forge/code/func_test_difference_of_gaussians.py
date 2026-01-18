import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage._shared.testing import assert_stacklevel
from skimage.filters import difference_of_gaussians, gaussian
@pytest.mark.parametrize('s', [1, (2, 3)])
@pytest.mark.parametrize('s2', [4, (5, 6)])
@pytest.mark.parametrize('channel_axis', [None, 0, 1, -1])
def test_difference_of_gaussians(s, s2, channel_axis):
    image = np.random.rand(10, 10)
    if channel_axis is not None:
        n_channels = 5
        image = np.stack((image,) * n_channels, channel_axis)
    im1 = gaussian(image, sigma=s, preserve_range=True, channel_axis=channel_axis)
    im2 = gaussian(image, sigma=s2, preserve_range=True, channel_axis=channel_axis)
    dog = im1 - im2
    dog2 = difference_of_gaussians(image, s, s2, channel_axis=channel_axis)
    assert np.allclose(dog, dog2)