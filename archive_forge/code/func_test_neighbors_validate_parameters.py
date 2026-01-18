import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('Estimator', [neighbors.KNeighborsClassifier, neighbors.RadiusNeighborsClassifier, neighbors.KNeighborsRegressor, neighbors.RadiusNeighborsRegressor])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_neighbors_validate_parameters(Estimator, csr_container):
    """Additional parameter validation for *Neighbors* estimators not covered by common
    validation."""
    X = rng.random_sample((10, 2))
    Xsparse = csr_container(X)
    X3 = rng.random_sample((10, 3))
    y = np.ones(10)
    nbrs = Estimator(algorithm='ball_tree', metric='haversine')
    msg = 'instance is not fitted yet'
    with pytest.raises(ValueError, match=msg):
        nbrs.predict(X)
    msg = "Metric 'haversine' not valid for sparse input."
    with pytest.raises(ValueError, match=msg):
        ignore_warnings(nbrs.fit(Xsparse, y))
    nbrs = Estimator(metric='haversine', algorithm='brute')
    nbrs.fit(X3, y)
    msg = 'Haversine distance only valid in 2 dimensions'
    with pytest.raises(ValueError, match=msg):
        nbrs.predict(X3)
    nbrs = Estimator()
    msg = re.escape('Found array with 0 sample(s)')
    with pytest.raises(ValueError, match=msg):
        nbrs.fit(np.ones((0, 2)), np.ones(0))
    msg = 'Found array with dim 3'
    with pytest.raises(ValueError, match=msg):
        nbrs.fit(X[:, :, None], y)
    nbrs.fit(X, y)
    msg = re.escape('Found array with 0 feature(s)')
    with pytest.raises(ValueError, match=msg):
        nbrs.predict([[]])