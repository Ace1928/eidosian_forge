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
@pytest.mark.parametrize('strategy', ['mean', 'median', 'most_frequent'])
def test_imputation_deletion_warning_feature_names(strategy):
    pd = pytest.importorskip('pandas')
    missing_values = np.nan
    feature_names = np.array(['a', 'b', 'c', 'd'], dtype=object)
    X = pd.DataFrame([[missing_values, missing_values, 1, missing_values], [4, missing_values, 2, 10]], columns=feature_names)
    imputer = SimpleImputer(strategy=strategy).fit(X)
    assert_array_equal(imputer.feature_names_in_, feature_names)
    with pytest.warns(UserWarning, match="Skipping features without any observed values: \\['b'\\]"):
        imputer.transform(X)