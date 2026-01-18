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
@pytest.mark.parametrize('Est', (HistGradientBoostingClassifier, HistGradientBoostingRegressor))
def test_categorical_spec_errors_with_feature_names(Est):
    pd = pytest.importorskip('pandas')
    n_samples = 10
    X = pd.DataFrame({'f0': range(n_samples), 'f1': range(n_samples), 'f2': [1.0] * n_samples})
    y = [0, 1] * (n_samples // 2)
    est = Est(categorical_features=['f0', 'f1', 'f3'])
    expected_msg = re.escape("categorical_features has a item value 'f3' which is not a valid feature name of the training data.")
    with pytest.raises(ValueError, match=expected_msg):
        est.fit(X, y)
    est = Est(categorical_features=['f0', 'f1'])
    expected_msg = re.escape('categorical_features should be passed as an array of integers or as a boolean mask when the model is fitted on data without feature names.')
    with pytest.raises(ValueError, match=expected_msg):
        est.fit(X.to_numpy(), y)