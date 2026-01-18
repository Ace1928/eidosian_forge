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
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_imputation_median_special_cases(csc_container):
    X = np.array([[0, np.nan, np.nan], [5, np.nan, np.nan], [0, 0, np.nan], [-5, 0, np.nan], [0, 5, np.nan], [4, 5, np.nan], [-4, -5, np.nan], [-1, 2, np.nan]]).transpose()
    X_imputed_median = np.array([[0, 0, 0], [5, 5, 5], [0, 0, 0], [-5, 0, -2.5], [0, 5, 2.5], [4, 5, 4.5], [-4, -5, -4.5], [-1, 2, 0.5]]).transpose()
    statistics_median = [0, 5, 0, -2.5, 2.5, 4.5, -4.5, 0.5]
    _check_statistics(X, X_imputed_median, 'median', statistics_median, np.nan, csc_container)