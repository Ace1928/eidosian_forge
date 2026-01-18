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
@pytest.mark.parametrize('method', ('online', 'batch'))
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_lda_perplexity(method, csr_container):
    n_components, X = _build_sparse_array(csr_container)
    lda_1 = LatentDirichletAllocation(n_components=n_components, max_iter=1, learning_method=method, total_samples=100, random_state=0)
    lda_2 = LatentDirichletAllocation(n_components=n_components, max_iter=10, learning_method=method, total_samples=100, random_state=0)
    lda_1.fit(X)
    perp_1 = lda_1.perplexity(X, sub_sampling=False)
    lda_2.fit(X)
    perp_2 = lda_2.perplexity(X, sub_sampling=False)
    assert perp_1 >= perp_2
    perp_1_subsampling = lda_1.perplexity(X, sub_sampling=True)
    perp_2_subsampling = lda_2.perplexity(X, sub_sampling=True)
    assert perp_1_subsampling >= perp_2_subsampling