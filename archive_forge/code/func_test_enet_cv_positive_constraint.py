import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_enet_cv_positive_constraint():
    X, y, X_test, y_test = build_dataset()
    max_iter = 500
    enetcv_unconstrained = ElasticNetCV(n_alphas=3, eps=0.1, max_iter=max_iter, cv=2, n_jobs=1)
    enetcv_unconstrained.fit(X, y)
    assert min(enetcv_unconstrained.coef_) < 0
    enetcv_constrained = ElasticNetCV(n_alphas=3, eps=0.1, max_iter=max_iter, cv=2, positive=True, n_jobs=1)
    enetcv_constrained.fit(X, y)
    assert min(enetcv_constrained.coef_) >= 0