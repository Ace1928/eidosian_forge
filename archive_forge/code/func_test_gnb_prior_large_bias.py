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
def test_gnb_prior_large_bias():
    """Test if good prediction when class prior favor largely one class"""
    clf = GaussianNB(priors=np.array([0.01, 0.99]))
    clf.fit(X, y)
    assert clf.predict([[-0.1, -0.1]]) == np.array([2])