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
@pytest.mark.parametrize('estimator', [ElasticNetCV, LassoCV])
def test_linear_models_cv_fit_with_loky(estimator):
    X, y = make_regression(int(1000000.0) // 8 + 1, 1)
    assert X.nbytes > 1000000.0
    with joblib.parallel_backend('loky'):
        estimator(n_jobs=2, cv=3).fit(X, y)