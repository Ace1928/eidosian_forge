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
@ignore_warnings(category=UserWarning)
@pytest.mark.parametrize(['Estimator', 'solver'], [[NMF, {'solver': 'cd'}], [NMF, {'solver': 'mu'}], [MiniBatchNMF, {}]])
@pytest.mark.parametrize('init', (None, 'nndsvd', 'nndsvda', 'nndsvdar', 'random'))
@pytest.mark.parametrize('alpha_W', (0.0, 1.0))
@pytest.mark.parametrize('alpha_H', (0.0, 1.0, 'same'))
def test_nmf_fit_nn_output(Estimator, solver, init, alpha_W, alpha_H):
    A = np.c_[5.0 - np.arange(1, 6), 5.0 + np.arange(1, 6)]
    model = Estimator(n_components=2, init=init, alpha_W=alpha_W, alpha_H=alpha_H, random_state=0, **solver)
    transf = model.fit_transform(A)
    assert not ((model.components_ < 0).any() or (transf < 0).any())