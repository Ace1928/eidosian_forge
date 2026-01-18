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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_missing_indicator_sparse_no_explicit_zeros(csr_container):
    X = csr_container([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    mi = MissingIndicator(features='all', missing_values=1)
    Xt = mi.fit_transform(X)
    assert Xt.getnnz() == Xt.sum()