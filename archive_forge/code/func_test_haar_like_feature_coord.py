from random import shuffle
import pytest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
@pytest.mark.parametrize('feature_type,height,width,expected_coord', [('type-2-x', 2, 2, [[[(0, 0), (0, 0)], [(0, 1), (0, 1)]], [[(0, 0), (1, 0)], [(0, 1), (1, 1)]], [[(1, 0), (1, 0)], [(1, 1), (1, 1)]]]), ('type-2-y', 2, 2, [[[(0, 0), (0, 0)], [(1, 0), (1, 0)]], [[(0, 0), (0, 1)], [(1, 0), (1, 1)]], [[(0, 1), (0, 1)], [(1, 1), (1, 1)]]]), ('type-3-x', 3, 3, [[[(0, 0), (0, 0)], [(0, 1), (0, 1)], [(0, 2), (0, 2)]], [[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 2), (1, 2)]], [[(0, 0), (2, 0)], [(0, 1), (2, 1)], [(0, 2), (2, 2)]], [[(1, 0), (1, 0)], [(1, 1), (1, 1)], [(1, 2), (1, 2)]], [[(1, 0), (2, 0)], [(1, 1), (2, 1)], [(1, 2), (2, 2)]], [[(2, 0), (2, 0)], [(2, 1), (2, 1)], [(2, 2), (2, 2)]]]), ('type-3-y', 3, 3, [[[(0, 0), (0, 0)], [(1, 0), (1, 0)], [(2, 0), (2, 0)]], [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(2, 0), (2, 1)]], [[(0, 0), (0, 2)], [(1, 0), (1, 2)], [(2, 0), (2, 2)]], [[(0, 1), (0, 1)], [(1, 1), (1, 1)], [(2, 1), (2, 1)]], [[(0, 1), (0, 2)], [(1, 1), (1, 2)], [(2, 1), (2, 2)]], [[(0, 2), (0, 2)], [(1, 2), (1, 2)], [(2, 2), (2, 2)]]]), ('type-4', 2, 2, [[[(0, 0), (0, 0)], [(0, 1), (0, 1)], [(1, 1), (1, 1)], [(1, 0), (1, 0)]]])])
def test_haar_like_feature_coord(feature_type, height, width, expected_coord):
    feat_coord, feat_type = haar_like_feature_coord(width, height, feature_type)
    feat_coord = np.array([hf for hf in feat_coord])
    assert_array_equal(feat_coord, expected_coord)
    assert np.all(feat_type == feature_type)