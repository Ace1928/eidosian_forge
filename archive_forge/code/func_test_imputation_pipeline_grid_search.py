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
def test_imputation_pipeline_grid_search():
    X = _sparse_random_matrix(100, 100, density=0.1)
    missing_values = X.data[0]
    pipeline = Pipeline([('imputer', SimpleImputer(missing_values=missing_values)), ('tree', tree.DecisionTreeRegressor(random_state=0))])
    parameters = {'imputer__strategy': ['mean', 'median', 'most_frequent']}
    Y = _sparse_random_matrix(100, 1, density=0.1).toarray()
    gs = GridSearchCV(pipeline, parameters)
    gs.fit(X, Y)