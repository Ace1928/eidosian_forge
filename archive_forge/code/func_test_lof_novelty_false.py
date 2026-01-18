import numpy as np
from sklearn.base import clone
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.neighbors import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import assert_array_almost_equal
def test_lof_novelty_false():
    n_neighbors = 4
    rng = np.random.RandomState(0)
    X = rng.randn(40, 2)
    est_chain = make_pipeline(KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance'), LocalOutlierFactor(metric='precomputed', n_neighbors=n_neighbors, novelty=False, contamination='auto'))
    est_compact = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False, contamination='auto')
    pred_chain = est_chain.fit_predict(X)
    pred_compact = est_compact.fit_predict(X)
    assert_array_almost_equal(pred_chain, pred_compact)