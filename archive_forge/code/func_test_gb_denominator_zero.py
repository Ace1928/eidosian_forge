import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._gb import _safe_divide
from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.svm import NuSVR
from sklearn.utils import check_random_state, tosequence
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_gb_denominator_zero(global_random_seed):
    """Test _update_terminal_regions denominator is not zero.

    For instance for log loss based binary classification, the line search step might
    become nan/inf as denominator = hessian = prob * (1 - prob) and prob = 0 or 1 can
    happen.
    Here, we create a situation were this happens (at least with roughly 80%) based
    on the random seed.
    """
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=20)
    params = {'learning_rate': 1.0, 'subsample': 0.5, 'n_estimators': 100, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': global_random_seed, 'min_samples_leaf': 2}
    clf = GradientBoostingClassifier(**params)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        clf.fit(X, y)