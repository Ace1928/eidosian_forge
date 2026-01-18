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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_lda_fit_batch(csr_container):
    rng = np.random.RandomState(0)
    n_components, X = _build_sparse_array(csr_container)
    lda = LatentDirichletAllocation(n_components=n_components, evaluate_every=1, learning_method='batch', random_state=rng)
    lda.fit(X)
    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for component in lda.components_:
        top_idx = set(component.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps