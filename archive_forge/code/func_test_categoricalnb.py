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
def test_categoricalnb(global_random_seed):
    clf = CategoricalNB()
    X2, y2 = get_random_integer_x_three_classes_y(global_random_seed)
    y_pred = clf.fit(X2, y2).predict(X2)
    assert_array_equal(y_pred, y2)
    X3 = np.array([[1, 4], [2, 5]])
    y3 = np.array([1, 2])
    clf = CategoricalNB(alpha=1, fit_prior=False)
    clf.fit(X3, y3)
    assert_array_equal(clf.n_categories_, np.array([3, 6]))
    X = np.array([[0, -1]])
    y = np.array([1])
    error_msg = re.escape('Negative values in data passed to CategoricalNB (input X)')
    with pytest.raises(ValueError, match=error_msg):
        clf.predict(X)
    with pytest.raises(ValueError, match=error_msg):
        clf.fit(X, y)
    X3_test = np.array([[2, 5]])
    bayes_numerator = np.array([[1 / 3 * 1 / 3, 2 / 3 * 2 / 3]])
    bayes_denominator = bayes_numerator.sum()
    assert_array_almost_equal(clf.predict_proba(X3_test), bayes_numerator / bayes_denominator)
    assert len(clf.category_count_) == X3.shape[1]
    X = np.array([[0, 0], [0, 1], [0, 0], [1, 1]])
    y = np.array([1, 1, 2, 2])
    clf = CategoricalNB(alpha=1, fit_prior=False)
    clf.fit(X, y)
    assert_array_equal(clf.predict(np.array([[0, 0]])), np.array([1]))
    assert_array_equal(clf.n_categories_, np.array([2, 2]))
    for factor in [1.0, 0.3, 5, 0.0001]:
        X = np.array([[0, 0], [0, 1], [0, 0], [1, 1]])
        y = np.array([1, 1, 2, 2])
        sample_weight = np.array([1, 1, 10, 0.1]) * factor
        clf = CategoricalNB(alpha=1, fit_prior=False)
        clf.fit(X, y, sample_weight=sample_weight)
        assert_array_equal(clf.predict(np.array([[0, 0]])), np.array([2]))
        assert_array_equal(clf.n_categories_, np.array([2, 2]))