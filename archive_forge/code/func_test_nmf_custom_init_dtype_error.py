import re
import sys
import warnings
from io import StringIO
import numpy as np
import pytest
from scipy import linalg
from sklearn.base import clone
from sklearn.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from sklearn.decomposition import _nmf as nmf  # For testing internals
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import (
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.filterwarnings('ignore:The default value of `n_components` will change')
@pytest.mark.parametrize('Estimator', [NMF, MiniBatchNMF])
def test_nmf_custom_init_dtype_error(Estimator):
    rng = np.random.RandomState(0)
    X = rng.random_sample((20, 15))
    H = rng.random_sample((15, 15)).astype(np.float32)
    W = rng.random_sample((20, 15))
    with pytest.raises(TypeError, match='should have the same dtype as X'):
        Estimator(init='custom').fit(X, H=H, W=W)
    with pytest.raises(TypeError, match='should have the same dtype as X'):
        non_negative_factorization(X, H=H, update_H=False)