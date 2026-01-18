import itertools
import os
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose
def test_fastica_whiten_unit_variance():
    """Test unit variance of transformed data using FastICA algorithm.

    Bug #13056
    """
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10))
    n_components = X.shape[1]
    ica = FastICA(n_components=n_components, whiten='unit-variance', random_state=0)
    Xt = ica.fit_transform(X)
    assert np.var(Xt) == pytest.approx(1.0)