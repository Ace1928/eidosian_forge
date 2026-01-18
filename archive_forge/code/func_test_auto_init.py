import re
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.optimize import check_grad
from sklearn import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
@pytest.mark.parametrize('n_samples', [3, 5, 7, 11])
@pytest.mark.parametrize('n_features', [3, 5, 7, 11])
@pytest.mark.parametrize('n_classes', [5, 7, 11])
@pytest.mark.parametrize('n_components', [3, 5, 7, 11])
def test_auto_init(n_samples, n_features, n_classes, n_components):
    rng = np.random.RandomState(42)
    nca_base = NeighborhoodComponentsAnalysis(init='auto', n_components=n_components, max_iter=1, random_state=rng)
    if n_classes >= n_samples:
        pass
    else:
        X = rng.randn(n_samples, n_features)
        y = np.tile(range(n_classes), n_samples // n_classes + 1)[:n_samples]
        if n_components > n_features:
            pass
        else:
            nca = clone(nca_base)
            nca.fit(X, y)
            if n_components <= min(n_classes - 1, n_features):
                nca_other = clone(nca_base).set_params(init='lda')
            elif n_components < min(n_features, n_samples):
                nca_other = clone(nca_base).set_params(init='pca')
            else:
                nca_other = clone(nca_base).set_params(init='identity')
            nca_other.fit(X, y)
            assert_array_almost_equal(nca.components_, nca_other.components_)