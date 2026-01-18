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
def test_class_weights():
    """High level test to check class_weights."""
    n_samples = 255
    n_features = 2
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=0)
    y_is_1 = y == 1
    clf = HistGradientBoostingClassifier(min_samples_leaf=2, random_state=0, max_depth=2)
    sample_weight = np.ones(shape=n_samples)
    sample_weight[y_is_1] = 3.0
    clf.fit(X, y, sample_weight=sample_weight)
    class_weight = {0: 1.0, 1: 3.0}
    clf_class_weighted = clone(clf).set_params(class_weight=class_weight)
    clf_class_weighted.fit(X, y)
    assert_allclose(clf.decision_function(X), clf_class_weighted.decision_function(X))
    clf.fit(X, y, sample_weight=sample_weight ** 2)
    clf_class_weighted.fit(X, y, sample_weight=sample_weight)
    assert_allclose(clf.decision_function(X), clf_class_weighted.decision_function(X))
    X_imb = np.concatenate((X[~y_is_1], X[y_is_1][:10]))
    y_imb = np.concatenate((y[~y_is_1], y[y_is_1][:10]))
    clf_balanced = clone(clf).set_params(class_weight='balanced')
    clf_balanced.fit(X_imb, y_imb)
    class_weight = y_imb.shape[0] / (2 * np.bincount(y_imb))
    sample_weight = class_weight[y_imb]
    clf_sample_weight = clone(clf).set_params(class_weight=None)
    clf_sample_weight.fit(X_imb, y_imb, sample_weight=sample_weight)
    assert_allclose(clf_balanced.decision_function(X_imb), clf_sample_weight.decision_function(X_imb))