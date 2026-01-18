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
def test_logistic_regression_solvers_multiclass():
    """Test solvers converge to the same result for multiclass problems."""
    X, y = make_classification(n_samples=20, n_features=20, n_informative=10, n_classes=3, random_state=0)
    tol = 1e-07
    params = dict(fit_intercept=False, tol=tol, random_state=42, multi_class='ovr')
    solver_max_iter = {'sag': 1000, 'saga': 10000}
    regressors = {solver: LogisticRegression(solver=solver, max_iter=solver_max_iter.get(solver, 100), **params).fit(X, y) for solver in SOLVERS}
    for solver_1, solver_2 in itertools.combinations(regressors, r=2):
        assert_allclose(regressors[solver_1].coef_, regressors[solver_2].coef_, rtol=0.005 if solver_2 == 'saga' else 0.001, err_msg=f'{solver_1} vs {solver_2}')