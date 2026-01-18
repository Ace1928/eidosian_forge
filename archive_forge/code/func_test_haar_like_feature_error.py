from random import shuffle
import pytest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
def test_haar_like_feature_error():
    img = np.ones((5, 5), dtype=np.float32)
    img_ii = integral_image(img)
    feature_type = 'unknown_type'
    with pytest.raises(ValueError):
        haar_like_feature(img_ii, 0, 0, 5, 5, feature_type=feature_type)
        haar_like_feature_coord(5, 5, feature_type=feature_type)
        draw_haar_like_feature(img, 0, 0, 5, 5, feature_type=feature_type)
    feat_coord, feat_type = haar_like_feature_coord(5, 5, 'type-2-x')
    with pytest.raises(ValueError):
        haar_like_feature(img_ii, 0, 0, 5, 5, feature_type=feat_type[:3], feature_coord=feat_coord)