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
def test_n_iter_no_change():
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]
    tol = 0.01
    max_iter = 3000
    for n_iter_no_change in [2, 5, 10, 50, 100]:
        clf = MLPClassifier(tol=tol, max_iter=max_iter, solver='sgd', n_iter_no_change=n_iter_no_change)
        clf.fit(X, y)
        assert clf._no_improvement_count == n_iter_no_change + 1
        assert max_iter > clf.n_iter_