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
def test_early_stopping_with_sample_weights(monkeypatch):
    """Check that sample weights is passed in to the scorer and _raw_predict is not
    called."""
    mock_scorer = Mock(side_effect=get_scorer('neg_median_absolute_error'))

    def mock_check_scoring(estimator, scoring):
        assert scoring == 'neg_median_absolute_error'
        return mock_scorer
    monkeypatch.setattr(sklearn.ensemble._hist_gradient_boosting.gradient_boosting, 'check_scoring', mock_check_scoring)
    X, y = make_regression(random_state=0)
    sample_weight = np.ones_like(y)
    hist = HistGradientBoostingRegressor(max_iter=2, early_stopping=True, random_state=0, scoring='neg_median_absolute_error')
    mock_raw_predict = Mock(side_effect=hist._raw_predict)
    hist._raw_predict = mock_raw_predict
    hist.fit(X, y, sample_weight=sample_weight)
    assert mock_raw_predict.call_count == 0
    assert mock_scorer.call_count == 6
    for arg_list in mock_scorer.call_args_list:
        assert 'sample_weight' in arg_list[1]