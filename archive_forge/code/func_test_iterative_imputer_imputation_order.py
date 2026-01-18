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
@pytest.mark.parametrize('imputation_order', ['random', 'roman', 'ascending', 'descending', 'arabic'])
def test_iterative_imputer_imputation_order(imputation_order):
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    max_iter = 2
    X = _sparse_random_matrix(n, d, density=0.1, random_state=rng).toarray()
    X[:, 0] = 1
    imputer = IterativeImputer(missing_values=0, max_iter=max_iter, n_nearest_features=5, sample_posterior=False, skip_complete=True, min_value=0, max_value=1, verbose=1, imputation_order=imputation_order, random_state=rng)
    imputer.fit_transform(X)
    ordered_idx = [i.feat_idx for i in imputer.imputation_sequence_]
    assert len(ordered_idx) // imputer.n_iter_ == imputer.n_features_with_missing_
    if imputation_order == 'roman':
        assert np.all(ordered_idx[:d - 1] == np.arange(1, d))
    elif imputation_order == 'arabic':
        assert np.all(ordered_idx[:d - 1] == np.arange(d - 1, 0, -1))
    elif imputation_order == 'random':
        ordered_idx_round_1 = ordered_idx[:d - 1]
        ordered_idx_round_2 = ordered_idx[d - 1:]
        assert ordered_idx_round_1 != ordered_idx_round_2
    elif 'ending' in imputation_order:
        assert len(ordered_idx) == max_iter * (d - 1)