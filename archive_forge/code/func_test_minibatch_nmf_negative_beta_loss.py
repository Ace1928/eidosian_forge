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
@pytest.mark.parametrize('beta_loss', [-0.5, 0.0])
def test_minibatch_nmf_negative_beta_loss(beta_loss):
    """Check that an error is raised if beta_loss < 0 and X contains zeros."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(6, 5))
    X[X < 0] = 0
    nmf = MiniBatchNMF(beta_loss=beta_loss, random_state=0)
    msg = 'When beta_loss <= 0 and X contains zeros, the solver may diverge.'
    with pytest.raises(ValueError, match=msg):
        nmf.fit(X)