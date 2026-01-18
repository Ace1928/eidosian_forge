import copy
import pickle
import warnings
import numpy as np
import pytest
from scipy.special import expit
import sklearn
from sklearn.datasets import make_regression
from sklearn.isotonic import (
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.validation import check_array
def test_isotonic_regression_sample_weight_not_overwritten():
    """Check that calling fitting function of isotonic regression will not
    overwrite `sample_weight`.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20508
    """
    X, y = make_regression(n_samples=10, n_features=1, random_state=41)
    sample_weight_original = np.ones_like(y)
    sample_weight_original[0] = 10
    sample_weight_fit = sample_weight_original.copy()
    isotonic_regression(y, sample_weight=sample_weight_fit)
    assert_allclose(sample_weight_fit, sample_weight_original)
    IsotonicRegression().fit(X, y, sample_weight=sample_weight_fit)
    assert_allclose(sample_weight_fit, sample_weight_original)