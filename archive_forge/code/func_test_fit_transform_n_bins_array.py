import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.parametrize('strategy, expected, sample_weight', [('uniform', [[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]], None), ('kmeans', [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 2, 2, 2]], None), ('quantile', [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]], None), ('quantile', [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]], [1, 1, 3, 1]), ('quantile', [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], [0, 1, 3, 1]), ('kmeans', [[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1], [1, 2, 2, 2]], [1, 0, 3, 1])])
@pytest.mark.filterwarnings('ignore:In version 1.5 onwards, subsample=200_000')
def test_fit_transform_n_bins_array(strategy, expected, sample_weight):
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode='ordinal', strategy=strategy).fit(X, sample_weight=sample_weight)
    assert_array_equal(expected, est.transform(X))
    n_features = np.array(X).shape[1]
    assert est.bin_edges_.shape == (n_features,)
    for bin_edges, n_bins in zip(est.bin_edges_, est.n_bins_):
        assert bin_edges.shape == (n_bins + 1,)