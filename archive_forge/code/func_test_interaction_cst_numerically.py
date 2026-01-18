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
def test_interaction_cst_numerically():
    """Check that interaction constraints have no forbidden interactions."""
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = rng.uniform(size=(n_samples, 2))
    y = np.hstack((X, 5 * X[:, [0]] * X[:, [1]])).sum(axis=1)
    est = HistGradientBoostingRegressor(random_state=42)
    est.fit(X, y)
    est_no_interactions = HistGradientBoostingRegressor(interaction_cst=[{0}, {1}], random_state=42)
    est_no_interactions.fit(X, y)
    delta = 0.25
    X_test = X[(X[:, 0] < 1 - delta) & (X[:, 1] < 1 - delta)]
    X_delta_d_0 = X_test + [delta, 0]
    X_delta_0_d = X_test + [0, delta]
    X_delta_d_d = X_test + [delta, delta]
    assert_allclose(est_no_interactions.predict(X_delta_d_d) + est_no_interactions.predict(X_test) - est_no_interactions.predict(X_delta_d_0) - est_no_interactions.predict(X_delta_0_d), 0, atol=1e-12)
    assert np.all(est.predict(X_delta_d_d) + est.predict(X_test) - est.predict(X_delta_d_0) - est.predict(X_delta_0_d) > 0.01)