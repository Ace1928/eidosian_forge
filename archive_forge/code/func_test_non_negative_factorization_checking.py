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
def test_non_negative_factorization_checking():
    A = np.ones((2, 2))
    nnmf = non_negative_factorization
    msg = re.escape('Negative values in data passed to NMF (input H)')
    with pytest.raises(ValueError, match=msg):
        nnmf(A, A, -A, 2, init='custom')
    msg = re.escape('Negative values in data passed to NMF (input W)')
    with pytest.raises(ValueError, match=msg):
        nnmf(A, -A, A, 2, init='custom')
    msg = re.escape('Array passed to NMF (input H) is full of zeros')
    with pytest.raises(ValueError, match=msg):
        nnmf(A, A, 0 * A, 2, init='custom')