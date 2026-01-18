import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_exactly_zero_info_score():
    for i in np.logspace(1, 4, 4).astype(int):
        labels_a, labels_b = (np.ones(i, dtype=int), np.arange(i, dtype=int))
        assert normalized_mutual_info_score(labels_a, labels_b) == pytest.approx(0.0)
        assert v_measure_score(labels_a, labels_b) == pytest.approx(0.0)
        assert adjusted_mutual_info_score(labels_a, labels_b) == pytest.approx(0.0)
        assert normalized_mutual_info_score(labels_a, labels_b) == pytest.approx(0.0)
        for method in ['min', 'geometric', 'arithmetic', 'max']:
            assert adjusted_mutual_info_score(labels_a, labels_b, average_method=method) == pytest.approx(0.0)
            assert normalized_mutual_info_score(labels_a, labels_b, average_method=method) == pytest.approx(0.0)