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
@pytest.mark.parametrize('initial_strategy', ['mean', 'median', 'most_frequent', 'constant'])
def test_iterative_imputer_keep_empty_features(initial_strategy):
    """Check the behaviour of the iterative imputer with different initial strategy
    and keeping empty features (i.e. features containing only missing values).
    """
    X = np.array([[1, np.nan, 2], [3, np.nan, np.nan]])
    imputer = IterativeImputer(initial_strategy=initial_strategy, keep_empty_features=True)
    X_imputed = imputer.fit_transform(X)
    assert_allclose(X_imputed[:, 1], 0)
    X_imputed = imputer.transform(X)
    assert_allclose(X_imputed[:, 1], 0)