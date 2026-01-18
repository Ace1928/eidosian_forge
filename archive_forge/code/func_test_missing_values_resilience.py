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
@pytest.mark.parametrize('problem', ('classification', 'regression'))
@pytest.mark.parametrize('missing_proportion, expected_min_score_classification, expected_min_score_regression', [(0.1, 0.97, 0.89), (0.2, 0.93, 0.81), (0.5, 0.79, 0.52)])
def test_missing_values_resilience(problem, missing_proportion, expected_min_score_classification, expected_min_score_regression):
    rng = np.random.RandomState(0)
    n_samples = 1000
    n_features = 2
    if problem == 'regression':
        X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features, random_state=rng)
        gb = HistGradientBoostingRegressor()
        expected_min_score = expected_min_score_regression
    else:
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, n_repeated=0, random_state=rng)
        gb = HistGradientBoostingClassifier()
        expected_min_score = expected_min_score_classification
    mask = rng.binomial(1, missing_proportion, size=X.shape).astype(bool)
    X[mask] = np.nan
    gb.fit(X, y)
    assert gb.score(X, y) > expected_min_score