import itertools
import os
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose
def test_fastica_eigh_low_rank_warning(global_random_seed):
    """Test FastICA eigh solver raises warning for low-rank data."""
    rng = np.random.RandomState(global_random_seed)
    A = rng.randn(10, 2)
    X = A @ A.T
    ica = FastICA(random_state=0, whiten='unit-variance', whiten_solver='eigh')
    msg = 'There are some small singular values'
    with pytest.warns(UserWarning, match=msg):
        ica.fit(X)