from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_bayesian_ridge_scores():
    """Check scores attribute shape"""
    X, y = (diabetes.data, diabetes.target)
    clf = BayesianRidge(compute_score=True)
    clf.fit(X, y)
    assert clf.scores_.shape == (clf.n_iter_ + 1,)