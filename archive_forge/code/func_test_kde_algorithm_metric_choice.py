import joblib
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors
from sklearn.neighbors._ball_tree import kernel_norm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('algorithm', ['auto', 'ball_tree', 'kd_tree'])
@pytest.mark.parametrize('metric', ['euclidean', 'minkowski', 'manhattan', 'chebyshev', 'haversine'])
def test_kde_algorithm_metric_choice(algorithm, metric):
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    Y = rng.randn(10, 2)
    kde = KernelDensity(algorithm=algorithm, metric=metric)
    if algorithm == 'kd_tree' and metric not in KDTree.valid_metrics:
        with pytest.raises(ValueError, match='invalid metric'):
            kde.fit(X)
    else:
        kde.fit(X)
        y_dens = kde.score_samples(Y)
        assert y_dens.shape == Y.shape[:1]