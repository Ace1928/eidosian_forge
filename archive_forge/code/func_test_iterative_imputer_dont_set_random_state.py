import io
import re
import warnings
from itertools import product
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import kstest
from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.impute._base import _most_frequent
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_union
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('rs_imputer', [None, 1, np.random.RandomState(seed=1)])
@pytest.mark.parametrize('rs_estimator', [None, 1, np.random.RandomState(seed=1)])
def test_iterative_imputer_dont_set_random_state(rs_imputer, rs_estimator):

    class ZeroEstimator:

        def __init__(self, random_state):
            self.random_state = random_state

        def fit(self, *args, **kgards):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0])
    estimator = ZeroEstimator(random_state=rs_estimator)
    imputer = IterativeImputer(random_state=rs_imputer)
    X_train = np.zeros((10, 3))
    imputer.fit(X_train)
    assert estimator.random_state == rs_estimator