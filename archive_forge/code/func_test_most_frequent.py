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
@pytest.mark.parametrize('expected,array,dtype,extra_value,n_repeat', [('extra_value', ['a', 'b', 'c'], object, 'extra_value', 2), ('most_frequent_value', ['most_frequent_value', 'most_frequent_value', 'value'], object, 'extra_value', 1), ('a', ['min_value', 'min_valuevalue'], object, 'a', 2), ('min_value', ['min_value', 'min_value', 'value'], object, 'z', 2), (10, [1, 2, 3], int, 10, 2), (1, [1, 1, 2], int, 10, 1), (10, [20, 20, 1], int, 10, 2), (1, [1, 1, 20], int, 10, 2)])
def test_most_frequent(expected, array, dtype, extra_value, n_repeat):
    assert expected == _most_frequent(np.array(array, dtype=dtype), extra_value, n_repeat)