import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_boundary_case_ch2():
    X = np.array([[10, 20], [20, 20], [20, 30]])
    y = np.array([[1], [0], [0]])
    scores, pvalues = chi2(X, y)
    assert_array_almost_equal(scores, np.array([4.0, 0.71428571]))
    assert_array_almost_equal(pvalues, np.array([0.04550026, 0.39802472]))
    filter_fdr = SelectFdr(chi2, alpha=0.1)
    filter_fdr.fit(X, y)
    support_fdr = filter_fdr.get_support()
    assert_array_equal(support_fdr, np.array([True, False]))
    filter_kbest = SelectKBest(chi2, k=1)
    filter_kbest.fit(X, y)
    support_kbest = filter_kbest.get_support()
    assert_array_equal(support_kbest, np.array([True, False]))
    filter_percentile = SelectPercentile(chi2, percentile=50)
    filter_percentile.fit(X, y)
    support_percentile = filter_percentile.get_support()
    assert_array_equal(support_percentile, np.array([True, False]))
    filter_fpr = SelectFpr(chi2, alpha=0.1)
    filter_fpr.fit(X, y)
    support_fpr = filter_fpr.get_support()
    assert_array_equal(support_fpr, np.array([True, False]))
    filter_fwe = SelectFwe(chi2, alpha=0.1)
    filter_fwe.fit(X, y)
    support_fwe = filter_fwe.get_support()
    assert_array_equal(support_fwe, np.array([True, False]))