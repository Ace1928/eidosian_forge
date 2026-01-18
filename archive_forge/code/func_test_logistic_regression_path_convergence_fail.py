import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
def test_logistic_regression_path_convergence_fail():
    rng = np.random.RandomState(0)
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = [1] * 100 + [-1] * 100
    Cs = [1000.0]
    with pytest.warns(ConvergenceWarning) as record:
        _logistic_regression_path(X, y, Cs=Cs, tol=0.0, max_iter=1, random_state=0, verbose=0)
    assert len(record) == 1
    warn_msg = record[0].message.args[0]
    assert 'lbfgs failed to converge' in warn_msg
    assert 'Increase the number of iterations' in warn_msg
    assert 'scale the data' in warn_msg
    assert 'linear_model.html#logistic-regression' in warn_msg