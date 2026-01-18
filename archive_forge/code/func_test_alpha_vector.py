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
def test_alpha_vector():
    X = np.array([[1, 0], [1, 1]])
    y = np.array([0, 1])
    alpha = np.array([1, 2])
    nb = MultinomialNB(alpha=alpha, force_alpha=False)
    nb.partial_fit(X, y, classes=[0, 1])
    feature_prob = np.array([[1 / 2, 1 / 2], [2 / 5, 3 / 5]])
    assert_array_almost_equal(nb.feature_log_prob_, np.log(feature_prob))
    prob = np.array([[5 / 9, 4 / 9], [25 / 49, 24 / 49]])
    assert_array_almost_equal(nb.predict_proba(X), prob)
    alpha = np.array([1.0, -0.1])
    m_nb = MultinomialNB(alpha=alpha, force_alpha=False)
    expected_msg = 'All values in alpha must be greater than 0.'
    with pytest.raises(ValueError, match=expected_msg):
        m_nb.fit(X, y)
    ALPHA_MIN = 1e-10
    alpha = np.array([ALPHA_MIN / 2, 0.5])
    m_nb = MultinomialNB(alpha=alpha, force_alpha=False)
    m_nb.partial_fit(X, y, classes=[0, 1])
    assert_array_almost_equal(m_nb._check_alpha(), [ALPHA_MIN, 0.5], decimal=12)
    alpha = np.array([1.0, 2.0, 3.0])
    m_nb = MultinomialNB(alpha=alpha, force_alpha=False)
    expected_msg = 'When alpha is an array, it should contains `n_features`'
    with pytest.raises(ValueError, match=expected_msg):
        m_nb.fit(X, y)