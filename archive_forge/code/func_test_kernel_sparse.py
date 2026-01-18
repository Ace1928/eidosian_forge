import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('kernel', (linear_kernel, polynomial_kernel, rbf_kernel, laplacian_kernel, sigmoid_kernel, cosine_similarity))
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_kernel_sparse(kernel, csr_container):
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    X_sparse = csr_container(X)
    K = kernel(X, X)
    K2 = kernel(X_sparse, X_sparse)
    assert_allclose(K, K2)