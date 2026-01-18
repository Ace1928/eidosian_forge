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
def test_nmf_custom_init_shape_error():
    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 5))
    H = rng.random_sample((2, 5))
    nmf = NMF(n_components=2, init='custom', random_state=0)
    with pytest.raises(ValueError, match='Array with wrong first dimension passed'):
        nmf.fit(X, H=H, W=rng.random_sample((5, 2)))
    with pytest.raises(ValueError, match='Array with wrong second dimension passed'):
        nmf.fit(X, H=H, W=rng.random_sample((6, 3)))