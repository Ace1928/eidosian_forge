import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._gb import _safe_divide
from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.svm import NuSVR
from sklearn.utils import check_random_state, tosequence
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_multinomial_error_exact_backward_compat():
    """Test multiclass log_loss GBT backward compat on a simple dataset.

    The results to compare against are taken from scikit-learn v1.2.0.
    """
    n_samples = 10
    y = np.arange(n_samples) % 4
    x1 = np.minimum(y, n_samples / 2)
    x2 = np.minimum(-y, -n_samples / 2)
    X = np.c_[x1, x2]
    gbt = GradientBoostingClassifier(loss='log_loss', n_estimators=100).fit(X, y)
    pred_result = np.array([[0.999999727, 1.11956255e-07, 8.04921671e-08, 8.04921668e-08], [1.11956254e-07, 0.999999727, 8.04921671e-08, 8.04921668e-08], [1.19417637e-07, 1.19417637e-07, 0.999999675, 8.60526098e-08], [1.19417637e-07, 1.19417637e-07, 8.60526088e-08, 0.999999675], [0.999999727, 1.11956255e-07, 8.04921671e-08, 8.04921668e-08], [1.11956254e-07, 0.999999727, 8.04921671e-08, 8.04921668e-08], [1.19417637e-07, 1.19417637e-07, 0.999999675, 8.60526098e-08], [1.19417637e-07, 1.19417637e-07, 8.60526088e-08, 0.999999675], [0.999999727, 1.11956255e-07, 8.04921671e-08, 8.04921668e-08], [1.11956254e-07, 0.999999727, 8.04921671e-08, 8.04921668e-08]])
    assert_allclose(gbt.predict_proba(X), pred_result, rtol=1e-08)
    train_score = np.array([1.1330015e-06, 9.75183397e-07, 8.39348103e-07, 7.22433588e-07, 6.21804338e-07, 5.35191943e-07, 4.60643966e-07, 3.9647993e-07, 3.41253434e-07, 2.9371955e-07])
    assert_allclose(gbt.train_score_[-10:], train_score, rtol=1e-08)