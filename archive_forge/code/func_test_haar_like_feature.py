from random import shuffle
import pytest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
@pytest.mark.parametrize('dtype', [np.uint8, np.int8, np.float32, np.float64])
@pytest.mark.parametrize('feature_type,shape_feature,expected_feature_value', [('type-2-x', (84,), [0.0]), ('type-2-y', (84,), [0.0]), ('type-3-x', (42,), [-5, -4.0, -3.0, -2.0, -1.0]), ('type-3-y', (42,), [-5, -4.0, -3.0, -2.0, -1.0]), ('type-4', (36,), [0.0])])
def test_haar_like_feature(feature_type, shape_feature, expected_feature_value, dtype):
    img = np.ones((5, 5), dtype=dtype)
    img_ii = integral_image(img)
    haar_feature = haar_like_feature(img_ii, 0, 0, 5, 5, feature_type=feature_type)
    assert_allclose(np.sort(np.unique(haar_feature)), expected_feature_value)