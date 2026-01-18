import sys
from io import StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.linalg import block_diag
from scipy.special import psi
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition._online_lda_fast import (
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_lda_transform():
    rng = np.random.RandomState(0)
    X = rng.randint(5, size=(20, 10))
    n_components = 3
    lda = LatentDirichletAllocation(n_components=n_components, random_state=rng)
    X_trans = lda.fit_transform(X)
    assert (X_trans > 0.0).any()
    assert_array_almost_equal(np.sum(X_trans, axis=1), np.ones(X_trans.shape[0]))