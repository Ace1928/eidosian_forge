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
def test_gnb_naive_bayes_scale_invariance():
    iris = load_iris()
    X, y = (iris.data, iris.target)
    labels = [GaussianNB().fit(f * X, y).predict(f * X) for f in [1e-10, 1, 10000000000.0]]
    assert_array_equal(labels[0], labels[1])
    assert_array_equal(labels[1], labels[2])