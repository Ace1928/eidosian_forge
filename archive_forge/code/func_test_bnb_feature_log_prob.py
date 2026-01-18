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
def test_bnb_feature_log_prob():
    X = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0]])
    Y = np.array([0, 0, 1, 2, 2])
    clf = BernoulliNB(alpha=1.0)
    clf.fit(X, Y)
    num = np.log(clf.feature_count_ + 1.0)
    denom = np.tile(np.log(clf.class_count_ + 2.0), (X.shape[1], 1)).T
    assert_array_almost_equal(clf.feature_log_prob_, num - denom)