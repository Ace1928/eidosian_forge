import copyreg
import io
import pickle
import re
import warnings
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
from joblib.numpy_pickle import NumpyPickler
from numpy.testing import assert_allclose, assert_array_equal
import sklearn
from sklearn._loss.loss import (
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_regressor
from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification, make_low_rank_matrix, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import get_scorer, mean_gamma_deviance, mean_poisson_deviance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder
from sklearn.utils import _IS_32BIT, shuffle
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import _convert_container
def test_categorical_encoding_strategies():
    rng = np.random.RandomState(0)
    n_samples = 10000
    f1 = rng.rand(n_samples)
    f2 = rng.randint(6, size=n_samples)
    X = np.c_[f1, f2]
    y = np.zeros(shape=n_samples)
    y[X[:, 1] % 2 == 0] = 1
    assert 0.49 < y.mean() < 0.51
    native_cat_specs = [[False, True], [1]]
    try:
        import pandas as pd
        X = pd.DataFrame(X, columns=['f_0', 'f_1'])
        native_cat_specs.append(['f_1'])
    except ImportError:
        pass
    for native_cat_spec in native_cat_specs:
        clf_cat = HistGradientBoostingClassifier(max_iter=1, max_depth=1, categorical_features=native_cat_spec)
        clf_cat.fit(X, y)
        assert cross_val_score(clf_cat, X, y).mean() == 1
    expected_left_bitset = [21, 0, 0, 0, 0, 0, 0, 0]
    left_bitset = clf_cat.fit(X, y)._predictors[0][0].raw_left_cat_bitsets[0]
    assert_array_equal(left_bitset, expected_left_bitset)
    clf_no_cat = HistGradientBoostingClassifier(max_iter=1, max_depth=4, categorical_features=None)
    assert cross_val_score(clf_no_cat, X, y).mean() < 0.9
    clf_no_cat.set_params(max_depth=5)
    assert cross_val_score(clf_no_cat, X, y).mean() == 1
    ct = make_column_transformer((OneHotEncoder(sparse_output=False), [1]), remainder='passthrough')
    X_ohe = ct.fit_transform(X)
    clf_no_cat.set_params(max_depth=2)
    assert cross_val_score(clf_no_cat, X_ohe, y).mean() < 0.9
    clf_no_cat.set_params(max_depth=3)
    assert cross_val_score(clf_no_cat, X_ohe, y).mean() == 1