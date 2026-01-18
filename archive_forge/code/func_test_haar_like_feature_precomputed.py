from random import shuffle
import pytest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
@pytest.mark.parametrize('feature_type', ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4', ['type-2-y', 'type-3-x', 'type-4']])
def test_haar_like_feature_precomputed(feature_type):
    img = np.ones((5, 5), dtype=np.int8)
    img_ii = integral_image(img)
    if isinstance(feature_type, list):
        shuffle(feature_type)
        feat_coord, feat_type = zip(*[haar_like_feature_coord(5, 5, feat_t) for feat_t in feature_type])
        feat_coord = np.concatenate(feat_coord)
        feat_type = np.concatenate(feat_type)
    else:
        feat_coord, feat_type = haar_like_feature_coord(5, 5, feature_type)
    haar_feature_precomputed = haar_like_feature(img_ii, 0, 0, 5, 5, feature_type=feat_type, feature_coord=feat_coord)
    haar_feature = haar_like_feature(img_ii, 0, 0, 5, 5, feature_type)
    assert_array_equal(haar_feature_precomputed, haar_feature)