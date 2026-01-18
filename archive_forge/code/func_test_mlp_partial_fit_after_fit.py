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
def test_mlp_partial_fit_after_fit(MLPEstimator):
    """Check partial fit does not fail after fit when early_stopping=True.

    Non-regression test for gh-25693.
    """
    mlp = MLPEstimator(early_stopping=True, random_state=0).fit(X_iris, y_iris)
    msg = 'partial_fit does not support early_stopping=True'
    with pytest.raises(ValueError, match=msg):
        mlp.partial_fit(X_iris, y_iris)