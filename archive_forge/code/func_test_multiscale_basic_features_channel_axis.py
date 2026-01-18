import pytest
import numpy as np
from skimage.feature import multiscale_basic_features
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2])
def test_multiscale_basic_features_channel_axis(channel_axis):
    num_channels = 5
    shape_spatial = (10, 10)
    ndim = len(shape_spatial)
    shape = tuple(np.insert(shape_spatial, channel_axis % (ndim + 1), num_channels))
    img = np.zeros(shape)
    img[:10] = 1
    img += 0.05 * np.random.randn(*img.shape)
    n_sigmas = 2
    features = multiscale_basic_features(img, sigma_min=1, sigma_max=2, channel_axis=channel_axis)
    assert features.shape[-1] == 5 * n_sigmas * 4
    assert features.shape[:-1] == np.moveaxis(img, channel_axis, -1).shape[:-1]
    features = multiscale_basic_features(img, sigma_min=1, sigma_max=2)
    assert features.shape[-1] == n_sigmas * 5
    assert features.shape[:-1] == img.shape