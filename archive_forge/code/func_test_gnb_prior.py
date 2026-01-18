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
def test_gnb_prior(global_random_seed):
    clf = GaussianNB().fit(X, y)
    assert_array_almost_equal(np.array([3, 3]) / 6.0, clf.class_prior_, 8)
    X1, y1 = get_random_normal_x_binary_y(global_random_seed)
    clf = GaussianNB().fit(X1, y1)
    assert_array_almost_equal(clf.class_prior_.sum(), 1)