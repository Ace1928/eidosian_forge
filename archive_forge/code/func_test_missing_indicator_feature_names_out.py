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
def test_missing_indicator_feature_names_out():
    """Check that missing indicator return the feature names with a prefix."""
    pd = pytest.importorskip('pandas')
    missing_values = np.nan
    X = pd.DataFrame([[missing_values, missing_values, 1, missing_values], [4, missing_values, 2, 10]], columns=['a', 'b', 'c', 'd'])
    indicator = MissingIndicator(missing_values=missing_values).fit(X)
    feature_names = indicator.get_feature_names_out()
    expected_names = ['missingindicator_a', 'missingindicator_b', 'missingindicator_d']
    assert_array_equal(expected_names, feature_names)