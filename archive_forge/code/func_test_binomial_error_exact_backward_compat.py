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
def test_binomial_error_exact_backward_compat():
    """Test binary log_loss GBT backward compat on a simple dataset.

    The results to compare against are taken from scikit-learn v1.2.0.
    """
    n_samples = 10
    y = np.arange(n_samples) % 2
    x1 = np.minimum(y, n_samples / 2)
    x2 = np.minimum(-y, -n_samples / 2)
    X = np.c_[x1, x2]
    gbt = GradientBoostingClassifier(loss='log_loss', n_estimators=100).fit(X, y)
    pred_result = np.array([[0.999978098, 2.19017313e-05], [2.19017313e-05, 0.999978098], [0.999978098, 2.19017313e-05], [2.19017313e-05, 0.999978098], [0.999978098, 2.19017313e-05], [2.19017313e-05, 0.999978098], [0.999978098, 2.19017313e-05], [2.19017313e-05, 0.999978098], [0.999978098, 2.19017313e-05], [2.19017313e-05, 0.999978098]])
    assert_allclose(gbt.predict_proba(X), pred_result, rtol=1e-08)
    train_score = np.array([0.00010774221, 9.74889078e-05, 8.82113863e-05, 7.98167784e-05, 7.22210566e-05, 6.53481907e-05, 5.91293869e-05, 5.35023988e-05, 4.84109045e-05, 4.38039423e-05])
    assert_allclose(gbt.train_score_[-10:], train_score, rtol=1e-08)