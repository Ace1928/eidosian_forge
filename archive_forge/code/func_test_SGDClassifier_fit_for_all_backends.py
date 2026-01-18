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
@pytest.mark.parametrize('backend', ['loky', 'multiprocessing', 'threading'])
def test_SGDClassifier_fit_for_all_backends(backend):
    random_state = np.random.RandomState(42)
    X = sp.random(500, 2000, density=0.02, format='csr', random_state=random_state)
    y = random_state.choice(20, 500)
    clf_sequential = SGDClassifier(max_iter=1000, n_jobs=1, random_state=42)
    clf_sequential.fit(X, y)
    clf_parallel = SGDClassifier(max_iter=1000, n_jobs=4, random_state=42)
    with joblib.parallel_backend(backend=backend):
        clf_parallel.fit(X, y)
    assert_array_almost_equal(clf_sequential.coef_, clf_parallel.coef_)