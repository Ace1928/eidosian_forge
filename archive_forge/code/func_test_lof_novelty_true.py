import numpy as np
from sklearn.base import clone
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.neighbors import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import assert_array_almost_equal
def test_lof_novelty_true():
    n_neighbors = 4
    rng = np.random.RandomState(0)
    X1 = rng.randn(40, 2)
    X2 = rng.randn(40, 2)
    est_chain = make_pipeline(KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance'), LocalOutlierFactor(metric='precomputed', n_neighbors=n_neighbors, novelty=True, contamination='auto'))
    est_compact = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination='auto')
    pred_chain = est_chain.fit(X1).predict(X2)
    pred_compact = est_compact.fit(X1).predict(X2)
    assert_array_almost_equal(pred_chain, pred_compact)