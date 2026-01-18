import numpy as np
from sklearn.base import clone
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.neighbors import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import assert_array_almost_equal
def test_dbscan():
    radius = 0.3
    n_clusters = 3
    X = generate_clustered_data(n_clusters=n_clusters)
    est_chain = make_pipeline(RadiusNeighborsTransformer(radius=radius, mode='distance'), DBSCAN(metric='precomputed', eps=radius))
    est_compact = DBSCAN(eps=radius)
    labels_chain = est_chain.fit_predict(X)
    labels_compact = est_compact.fit_predict(X)
    assert_array_almost_equal(labels_chain, labels_compact)