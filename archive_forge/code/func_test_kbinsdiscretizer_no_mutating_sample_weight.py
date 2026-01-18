import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.filterwarnings('ignore:In version 1.5 onwards, subsample=200_000')
@pytest.mark.parametrize('strategy', ['kmeans', 'quantile'])
def test_kbinsdiscretizer_no_mutating_sample_weight(strategy):
    """Make sure that `sample_weight` is not changed in place."""
    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy=strategy)
    sample_weight = np.array([1, 3, 1, 2], dtype=np.float64)
    sample_weight_copy = np.copy(sample_weight)
    est.fit(X, sample_weight=sample_weight)
    assert_allclose(sample_weight, sample_weight_copy)