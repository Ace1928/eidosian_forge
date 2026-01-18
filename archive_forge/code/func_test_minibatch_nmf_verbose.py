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
def test_minibatch_nmf_verbose():
    A = np.random.RandomState(0).random_sample((100, 10))
    nmf = MiniBatchNMF(tol=0.01, random_state=0, verbose=1)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        nmf.fit(A)
    finally:
        sys.stdout = old_stdout