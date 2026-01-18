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
@pytest.mark.parametrize('Estimator', [NMF, MiniBatchNMF])
def test_n_components_greater_n_features(Estimator):
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(30, 10))
    Estimator(n_components=15, random_state=0, tol=0.01).fit(A)