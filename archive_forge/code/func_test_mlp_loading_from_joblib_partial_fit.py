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
def test_mlp_loading_from_joblib_partial_fit(tmp_path):
    """Loading from MLP and partial fitting updates weights. Non-regression
    test for #19626."""
    pre_trained_estimator = MLPRegressor(hidden_layer_sizes=(42,), random_state=42, learning_rate_init=0.01, max_iter=200)
    features, target = ([[2]], [4])
    pre_trained_estimator.fit(features, target)
    pickled_file = tmp_path / 'mlp.pkl'
    joblib.dump(pre_trained_estimator, pickled_file)
    load_estimator = joblib.load(pickled_file)
    fine_tune_features, fine_tune_target = ([[2]], [1])
    for _ in range(200):
        load_estimator.partial_fit(fine_tune_features, fine_tune_target)
    predicted_value = load_estimator.predict(fine_tune_features)
    assert_allclose(predicted_value, fine_tune_target, rtol=0.0001)