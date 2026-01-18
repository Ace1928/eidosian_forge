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
@pytest.mark.parametrize('dataframe_lib', ['pandas', 'polars'])
def test_categorical_different_order_same_model(dataframe_lib):
    """Check that the order of the categorical gives same model."""
    pytest.importorskip(dataframe_lib)
    rng = np.random.RandomState(42)
    n_samples = 1000
    f_ints = rng.randint(low=0, high=2, size=n_samples)
    y = f_ints.copy()
    flipped = rng.choice([True, False], size=n_samples, p=[0.1, 0.9])
    y[flipped] = 1 - y[flipped]
    f_cat_a_b = np.asarray(['A', 'B'])[f_ints]
    f_cat_b_a = np.asarray(['B', 'A'])[f_ints]
    df_a_b = _convert_container(f_cat_a_b[:, None], dataframe_lib, ['f_cat'], categorical_feature_names=['f_cat'])
    df_b_a = _convert_container(f_cat_b_a[:, None], dataframe_lib, ['f_cat'], categorical_feature_names=['f_cat'])
    hist_a_b = HistGradientBoostingClassifier(categorical_features='from_dtype', random_state=0)
    hist_b_a = HistGradientBoostingClassifier(categorical_features='from_dtype', random_state=0)
    hist_a_b.fit(df_a_b, y)
    hist_b_a.fit(df_b_a, y)
    assert len(hist_a_b._predictors) == len(hist_b_a._predictors)
    for predictor_1, predictor_2 in zip(hist_a_b._predictors, hist_b_a._predictors):
        assert len(predictor_1[0].nodes) == len(predictor_2[0].nodes)