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
def test_loss_epsilon_insensitive():
    loss = sgd_fast.EpsilonInsensitive(0.1)
    cases = [(0.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0, 0.0), (-2.05, -2.0, 0.0, 0.0), (3.05, 3.0, 0.0, 0.0), (2.2, 2.0, 0.1, 1.0), (2.0, -1.0, 2.9, 1.0), (2.0, 2.2, 0.1, -1.0), (-2.0, 1.0, 2.9, -1.0)]
    _test_loss_common(loss, cases)