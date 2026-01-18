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
def test_gnb_partial_fit():
    clf = GaussianNB().fit(X, y)
    clf_pf = GaussianNB().partial_fit(X, y, np.unique(y))
    assert_array_almost_equal(clf.theta_, clf_pf.theta_)
    assert_array_almost_equal(clf.var_, clf_pf.var_)
    assert_array_almost_equal(clf.class_prior_, clf_pf.class_prior_)
    clf_pf2 = GaussianNB().partial_fit(X[0::2, :], y[0::2], np.unique(y))
    clf_pf2.partial_fit(X[1::2], y[1::2])
    assert_array_almost_equal(clf.theta_, clf_pf2.theta_)
    assert_array_almost_equal(clf.var_, clf_pf2.var_)
    assert_array_almost_equal(clf.class_prior_, clf_pf2.class_prior_)