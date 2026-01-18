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
def test_lda_default_prior_params(csr_container):
    n_components, X = _build_sparse_array(csr_container)
    prior = 1.0 / n_components
    lda_1 = LatentDirichletAllocation(n_components=n_components, doc_topic_prior=prior, topic_word_prior=prior, random_state=0)
    lda_2 = LatentDirichletAllocation(n_components=n_components, random_state=0)
    topic_distr_1 = lda_1.fit_transform(X)
    topic_distr_2 = lda_2.fit_transform(X)
    assert_almost_equal(topic_distr_1, topic_distr_2)