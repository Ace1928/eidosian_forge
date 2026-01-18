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
def test_multi_core_gridsearch_and_early_stopping():
    param_grid = {'alpha': np.logspace(-4, 4, 9), 'n_iter_no_change': [5, 10, 50]}
    clf = SGDClassifier(tol=0.01, max_iter=1000, early_stopping=True, random_state=0)
    search = RandomizedSearchCV(clf, param_grid, n_iter=5, n_jobs=2, random_state=0)
    search.fit(iris.data, iris.target)
    assert search.best_score_ > 0.8