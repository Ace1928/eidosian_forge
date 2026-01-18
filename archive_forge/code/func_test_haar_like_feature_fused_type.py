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
@pytest.mark.parametrize('feature_type', ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4'])
def test_haar_like_feature_fused_type(dtype, feature_type):
    img = np.ones((5, 5), dtype=dtype)
    img_ii = integral_image(img)
    expected_dtype = img_ii.dtype
    if 'uint' in expected_dtype.name:
        expected_dtype = np.dtype(expected_dtype.name.replace('u', ''))
    haar_feature = haar_like_feature(img_ii, 0, 0, 5, 5, feature_type=feature_type)
    assert haar_feature.dtype == expected_dtype