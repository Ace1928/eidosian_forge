import re
import sys
from io import StringIO
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import assert_all_finite
def test_convergence_dtype_consistency():
    X_64 = Xdigits[:100].astype(np.float64)
    rbm_64 = BernoulliRBM(n_components=16, batch_size=5, n_iter=5, random_state=42)
    Xt_64 = rbm_64.fit_transform(X_64)
    X_32 = Xdigits[:100].astype(np.float32)
    rbm_32 = BernoulliRBM(n_components=16, batch_size=5, n_iter=5, random_state=42)
    Xt_32 = rbm_32.fit_transform(X_32)
    assert_allclose(Xt_64, Xt_32, rtol=1e-06, atol=0)
    assert_allclose(rbm_64.intercept_hidden_, rbm_32.intercept_hidden_, rtol=1e-06, atol=0)
    assert_allclose(rbm_64.intercept_visible_, rbm_32.intercept_visible_, rtol=1e-05, atol=0)
    assert_allclose(rbm_64.components_, rbm_32.components_, rtol=0.001, atol=0)
    assert_allclose(rbm_64.h_samples_, rbm_32.h_samples_)