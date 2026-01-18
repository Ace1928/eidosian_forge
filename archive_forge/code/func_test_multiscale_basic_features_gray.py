import pytest
import numpy as np
from skimage.feature import multiscale_basic_features
@pytest.mark.parametrize('edges', (False, True))
@pytest.mark.parametrize('texture', (False, True))
def test_multiscale_basic_features_gray(edges, texture):
    img = np.zeros((20, 20))
    img[:10] = 1
    img += 0.05 * np.random.randn(*img.shape)
    features = multiscale_basic_features(img, edges=edges, texture=texture)
    n_sigmas = 6
    intensity = True
    assert features.shape[-1] == n_sigmas * (int(intensity) + int(edges) + 2 * int(texture))
    assert features.shape[:-1] == img.shape[:]