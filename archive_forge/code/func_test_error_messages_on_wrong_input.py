import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_error_messages_on_wrong_input():
    for score_func in score_funcs:
        expected = 'Found input variables with inconsistent numbers of samples: \\[2, 3\\]'
        with pytest.raises(ValueError, match=expected):
            score_func([0, 1], [1, 1, 1])
        expected = 'labels_true must be 1D: shape is \\(2'
        with pytest.raises(ValueError, match=expected):
            score_func([[0, 1], [1, 0]], [1, 1, 1])
        expected = 'labels_pred must be 1D: shape is \\(2'
        with pytest.raises(ValueError, match=expected):
            score_func([0, 1, 0], [[1, 1], [0, 0]])