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
def test_gnb():
    clf = GaussianNB()
    y_pred = clf.fit(X, y).predict(X)
    assert_array_equal(y_pred, y)
    y_pred_proba = clf.predict_proba(X)
    y_pred_log_proba = clf.predict_log_proba(X)
    assert_array_almost_equal(np.log(y_pred_proba), y_pred_log_proba, 8)
    with pytest.raises(ValueError, match='The target label.* in y do not exist in the initial classes'):
        GaussianNB().partial_fit(X, y, classes=[0, 1])