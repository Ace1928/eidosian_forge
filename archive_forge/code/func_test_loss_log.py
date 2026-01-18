import pickle
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils._testing import (
def test_loss_log():
    loss = sgd_fast.Log()
    cases = [(1.0, 1.0, np.log(1.0 + np.exp(-1.0)), -1.0 / (np.exp(1.0) + 1.0)), (1.0, -1.0, np.log(1.0 + np.exp(1.0)), 1.0 / (np.exp(-1.0) + 1.0)), (-1.0, -1.0, np.log(1.0 + np.exp(-1.0)), 1.0 / (np.exp(1.0) + 1.0)), (-1.0, 1.0, np.log(1.0 + np.exp(1.0)), -1.0 / (np.exp(-1.0) + 1.0)), (0.0, 1.0, np.log(2), -0.5), (0.0, -1.0, np.log(2), 0.5), (17.9, -1.0, 17.9, 1.0), (-17.9, 1.0, 17.9, -1.0)]
    _test_loss_common(loss, cases)
    assert_almost_equal(loss.py_dloss(18.1, 1.0), np.exp(-18.1) * -1.0, 16)
    assert_almost_equal(loss.py_loss(18.1, 1.0), np.exp(-18.1), 16)
    assert_almost_equal(loss.py_dloss(-18.1, -1.0), np.exp(-18.1) * 1.0, 16)
    assert_almost_equal(loss.py_loss(-18.1, 1.0), 18.1, 16)