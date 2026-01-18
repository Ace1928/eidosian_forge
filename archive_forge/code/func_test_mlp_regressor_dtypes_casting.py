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
def test_mlp_regressor_dtypes_casting():
    mlp_64 = MLPRegressor(alpha=1e-05, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50)
    mlp_64.fit(X_digits[:300], y_digits[:300])
    pred_64 = mlp_64.predict(X_digits[300:])
    mlp_32 = MLPRegressor(alpha=1e-05, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50)
    mlp_32.fit(X_digits[:300].astype(np.float32), y_digits[:300])
    pred_32 = mlp_32.predict(X_digits[300:].astype(np.float32))
    assert_allclose(pred_64, pred_32, rtol=0.0001)