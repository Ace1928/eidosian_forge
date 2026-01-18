from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_bayesian_ridge_score_values():
    """Check value of score on toy example.

    Compute log marginal likelihood with equation (36) in Sparse Bayesian
    Learning and the Relevance Vector Machine (Tipping, 2001):

    - 0.5 * (log |Id/alpha + X.X^T/lambda| +
             y^T.(Id/alpha + X.X^T/lambda).y + n * log(2 * pi))
    + lambda_1 * log(lambda) - lambda_2 * lambda
    + alpha_1 * log(alpha) - alpha_2 * alpha

    and check equality with the score computed during training.
    """
    X, y = (diabetes.data, diabetes.target)
    n_samples = X.shape[0]
    eps = np.finfo(np.float64).eps
    alpha_ = 1.0 / (np.var(y) + eps)
    lambda_ = 1.0
    alpha_1 = 0.1
    alpha_2 = 0.1
    lambda_1 = 0.1
    lambda_2 = 0.1
    score = lambda_1 * log(lambda_) - lambda_2 * lambda_
    score += alpha_1 * log(alpha_) - alpha_2 * alpha_
    M = 1.0 / alpha_ * np.eye(n_samples) + 1.0 / lambda_ * np.dot(X, X.T)
    M_inv_dot_y = np.linalg.solve(M, y)
    score += -0.5 * (fast_logdet(M) + np.dot(y.T, M_inv_dot_y) + n_samples * log(2 * np.pi))
    clf = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2, max_iter=1, fit_intercept=False, compute_score=True)
    clf.fit(X, y)
    assert_almost_equal(clf.scores_[0], score, decimal=9)