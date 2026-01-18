import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS
def test_compute_class_weight_invariance():
    X, y = make_blobs(centers=2, random_state=0)
    X_1 = np.vstack([X] + [X[y == 1]] * 2)
    y_1 = np.hstack([y] + [y[y == 1]] * 2)
    X_0 = np.vstack([X] + [X[y == 0]] * 2)
    y_0 = np.hstack([y] + [y[y == 0]] * 2)
    X_ = np.vstack([X] * 2)
    y_ = np.hstack([y] * 2)
    logreg1 = LogisticRegression(class_weight='balanced').fit(X_1, y_1)
    logreg0 = LogisticRegression(class_weight='balanced').fit(X_0, y_0)
    logreg = LogisticRegression(class_weight='balanced').fit(X_, y_)
    assert_array_almost_equal(logreg1.coef_, logreg0.coef_)
    assert_array_almost_equal(logreg.coef_, logreg0.coef_)