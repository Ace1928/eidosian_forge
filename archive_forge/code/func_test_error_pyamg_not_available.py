from unittest.mock import Mock
import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import SpectralEmbedding, _spectral_embedding, spectral_embedding
from sklearn.manifold._spectral_embedding import (
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.fixes import (
from sklearn.utils.fixes import laplacian as csgraph_laplacian
@pytest.mark.skipif(pyamg_available, reason='PyAMG is installed and we should not test for an error.')
def test_error_pyamg_not_available():
    se_precomp = SpectralEmbedding(n_components=2, affinity='rbf', eigen_solver='amg')
    err_msg = "The eigen_solver was set to 'amg', but pyamg is not available."
    with pytest.raises(ValueError, match=err_msg):
        se_precomp.fit_transform(S)