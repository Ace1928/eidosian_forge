import numpy as np
from sklearn.base import clone
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.neighbors import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import assert_array_almost_equal
def test_isomap():
    algorithm = 'auto'
    n_neighbors = 10
    X, _ = make_blobs(random_state=0)
    X2, _ = make_blobs(random_state=1)
    est_chain = make_pipeline(KNeighborsTransformer(n_neighbors=n_neighbors, algorithm=algorithm, mode='distance'), Isomap(n_neighbors=n_neighbors, metric='precomputed'))
    est_compact = Isomap(n_neighbors=n_neighbors, neighbors_algorithm=algorithm)
    Xt_chain = est_chain.fit_transform(X)
    Xt_compact = est_compact.fit_transform(X)
    assert_array_almost_equal(Xt_chain, Xt_compact)
    Xt_chain = est_chain.transform(X2)
    Xt_compact = est_compact.transform(X2)
    assert_array_almost_equal(Xt_chain, Xt_compact)