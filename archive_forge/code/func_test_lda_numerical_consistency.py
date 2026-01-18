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
@pytest.mark.parametrize('learning_method', ('batch', 'online'))
def test_lda_numerical_consistency(learning_method, global_random_seed):
    """Check numerical consistency between np.float32 and np.float64."""
    rng = np.random.RandomState(global_random_seed)
    X64 = rng.uniform(size=(20, 10))
    X32 = X64.astype(np.float32)
    lda_64 = LatentDirichletAllocation(n_components=5, random_state=global_random_seed, learning_method=learning_method).fit(X64)
    lda_32 = LatentDirichletAllocation(n_components=5, random_state=global_random_seed, learning_method=learning_method).fit(X32)
    assert_allclose(lda_32.components_, lda_64.components_)
    assert_allclose(lda_32.transform(X32), lda_64.transform(X64))