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
def test_bnb():
    X = np.array([[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1]])
    Y = np.array([0, 0, 0, 1])
    clf = BernoulliNB(alpha=1.0)
    clf.fit(X, Y)
    class_prior = np.array([0.75, 0.25])
    assert_array_almost_equal(np.exp(clf.class_log_prior_), class_prior)
    feature_prob = np.array([[0.4, 0.8, 0.2, 0.4, 0.4, 0.2], [1 / 3.0, 2 / 3.0, 2 / 3.0, 1 / 3.0, 1 / 3.0, 2 / 3.0]])
    assert_array_almost_equal(np.exp(clf.feature_log_prob_), feature_prob)
    X_test = np.array([[0, 1, 1, 0, 0, 1]])
    unnorm_predict_proba = np.array([[0.005183999999999999, 0.02194787379972565]])
    predict_proba = unnorm_predict_proba / np.sum(unnorm_predict_proba)
    assert_array_almost_equal(clf.predict_proba(X_test), predict_proba)