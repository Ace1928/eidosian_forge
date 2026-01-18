from random import shuffle
import pytest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
@pytest.mark.parametrize('max_n_features,nnz_values', [(None, 46), (1, 4)])
def test_draw_haar_like_feature(max_n_features, nnz_values):
    img = np.zeros((5, 5), dtype=np.float32)
    coord, _ = haar_like_feature_coord(5, 5, 'type-4')
    image = draw_haar_like_feature(img, 0, 0, 5, 5, coord, max_n_features=max_n_features, rng=0)
    draw_haar_like_feature(img, 0, 0, 5, 5, coord, max_n_features=max_n_features, rng=0)
    assert image.shape == (5, 5, 3)
    assert np.count_nonzero(image) == nnz_values