import re
import warnings
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_gnb_sample_weight(global_random_seed):
    """Test whether sample weights are properly used in GNB."""
    sw = np.ones(6)
    clf = GaussianNB().fit(X, y)
    clf_sw = GaussianNB().fit(X, y, sw)
    assert_array_almost_equal(clf.theta_, clf_sw.theta_)
    assert_array_almost_equal(clf.var_, clf_sw.var_)
    rng = np.random.RandomState(global_random_seed)
    sw = rng.rand(y.shape[0])
    clf1 = GaussianNB().fit(X, y, sample_weight=sw)
    clf2 = GaussianNB().partial_fit(X, y, classes=[1, 2], sample_weight=sw / 2)
    clf2.partial_fit(X, y, sample_weight=sw / 2)
    assert_array_almost_equal(clf1.theta_, clf2.theta_)
    assert_array_almost_equal(clf1.var_, clf2.var_)
    ind = rng.randint(0, X.shape[0], 20)
    sample_weight = np.bincount(ind, minlength=X.shape[0])
    clf_dupl = GaussianNB().fit(X[ind], y[ind])
    clf_sw = GaussianNB().fit(X, y, sample_weight)
    assert_array_almost_equal(clf_dupl.theta_, clf_sw.theta_)
    assert_array_almost_equal(clf_dupl.var_, clf_sw.var_)
    sample_weight = (y == 1).astype(np.float64)
    clf = GaussianNB().fit(X, y, sample_weight=sample_weight)