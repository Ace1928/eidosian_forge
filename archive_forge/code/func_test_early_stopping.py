import re
import sys
import warnings
from io import StringIO
import joblib
import numpy as np
import pytest
from numpy.testing import (
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, scale
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('MLPEstimator', [MLPClassifier, MLPRegressor])
def test_early_stopping(MLPEstimator):
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]
    tol = 0.2
    mlp_estimator = MLPEstimator(tol=tol, max_iter=3000, solver='sgd', early_stopping=True)
    mlp_estimator.fit(X, y)
    assert mlp_estimator.max_iter > mlp_estimator.n_iter_
    assert mlp_estimator.best_loss_ is None
    assert isinstance(mlp_estimator.validation_scores_, list)
    valid_scores = mlp_estimator.validation_scores_
    best_valid_score = mlp_estimator.best_validation_score_
    assert max(valid_scores) == best_valid_score
    assert best_valid_score + tol > valid_scores[-2]
    assert best_valid_score + tol > valid_scores[-1]
    mlp_estimator = MLPEstimator(tol=tol, max_iter=3000, solver='sgd', early_stopping=False)
    mlp_estimator.fit(X, y)
    assert mlp_estimator.validation_scores_ is None
    assert mlp_estimator.best_validation_score_ is None
    assert mlp_estimator.best_loss_ is not None