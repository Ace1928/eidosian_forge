import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_adjusted_rand_score_overflow():
    """Check that large amount of data will not lead to overflow in
    `adjusted_rand_score`.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20305
    """
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 100000, dtype=np.int8)
    y_pred = rng.randint(0, 2, 100000, dtype=np.int8)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        adjusted_rand_score(y_true, y_pred)