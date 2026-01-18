import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_adjusted_mutual_info_score():
    labels_a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    labels_b = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2])
    mi = mutual_info_score(labels_a, labels_b)
    assert_almost_equal(mi, 0.41022, 5)
    C = contingency_matrix(labels_a, labels_b, sparse=True)
    mi = mutual_info_score(labels_a, labels_b, contingency=C)
    assert_almost_equal(mi, 0.41022, 5)
    C = contingency_matrix(labels_a, labels_b)
    mi = mutual_info_score(labels_a, labels_b, contingency=C)
    assert_almost_equal(mi, 0.41022, 5)
    n_samples = C.sum()
    emi = expected_mutual_information(C, n_samples)
    assert_almost_equal(emi, 0.15042, 5)
    ami = adjusted_mutual_info_score(labels_a, labels_b)
    assert_almost_equal(ami, 0.27821, 5)
    ami = adjusted_mutual_info_score([1, 1, 2, 2], [2, 2, 3, 3])
    assert ami == pytest.approx(1.0)
    a110 = np.array([list(labels_a) * 110]).flatten()
    b110 = np.array([list(labels_b) * 110]).flatten()
    ami = adjusted_mutual_info_score(a110, b110)
    assert_almost_equal(ami, 0.38, 2)