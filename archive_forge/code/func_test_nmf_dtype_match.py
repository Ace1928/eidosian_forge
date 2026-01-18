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
@pytest.mark.parametrize('dtype_in, dtype_out', [(np.float32, np.float32), (np.float64, np.float64), (np.int32, np.float64), (np.int64, np.float64)])
@pytest.mark.parametrize(['Estimator', 'solver'], [[NMF, {'solver': 'cd'}], [NMF, {'solver': 'mu'}], [MiniBatchNMF, {}]])
def test_nmf_dtype_match(Estimator, solver, dtype_in, dtype_out):
    X = np.random.RandomState(0).randn(20, 15).astype(dtype_in, copy=False)
    np.abs(X, out=X)
    nmf = Estimator(alpha_W=1.0, alpha_H=1.0, tol=0.01, random_state=0, **solver)
    assert nmf.fit(X).transform(X).dtype == dtype_out
    assert nmf.fit_transform(X).dtype == dtype_out
    assert nmf.components_.dtype == dtype_out