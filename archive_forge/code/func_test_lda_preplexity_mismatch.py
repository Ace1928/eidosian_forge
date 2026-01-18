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
def test_lda_preplexity_mismatch():
    rng = np.random.RandomState(0)
    n_components = rng.randint(3, 6)
    n_samples = rng.randint(6, 10)
    X = np.random.randint(4, size=(n_samples, 10))
    lda = LatentDirichletAllocation(n_components=n_components, learning_offset=5.0, total_samples=20, random_state=rng)
    lda.fit(X)
    invalid_n_samples = rng.randint(4, size=(n_samples + 1, n_components))
    with pytest.raises(ValueError, match='Number of samples'):
        lda._perplexity_precomp_distr(X, invalid_n_samples)
    invalid_n_components = rng.randint(4, size=(n_samples, n_components + 1))
    with pytest.raises(ValueError, match='Number of topics'):
        lda._perplexity_precomp_distr(X, invalid_n_components)