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
def test_kde_sample_weights_error():
    kde = KernelDensity()
    with pytest.raises(ValueError):
        kde.fit(np.random.random((200, 10)), sample_weight=np.random.random((200, 10)))
    with pytest.raises(ValueError):
        kde.fit(np.random.random((200, 10)), sample_weight=-np.random.random(200))