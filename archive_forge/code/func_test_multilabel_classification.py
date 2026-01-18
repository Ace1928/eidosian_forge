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
def test_multilabel_classification():
    X, y = make_multilabel_classification(n_samples=50, random_state=0, return_indicator=True)
    mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=50, alpha=1e-05, max_iter=150, random_state=0, activation='logistic', learning_rate_init=0.2)
    mlp.fit(X, y)
    assert mlp.score(X, y) > 0.97
    mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=50, max_iter=150, random_state=0, activation='logistic', alpha=1e-05, learning_rate_init=0.2)
    for i in range(100):
        mlp.partial_fit(X, y, classes=[0, 1, 2, 3, 4])
    assert mlp.score(X, y) > 0.9
    mlp = MLPClassifier(early_stopping=True)
    mlp.fit(X, y).predict(X)