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
@pytest.mark.parametrize('backend', ['threading', 'loky'])
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_knn_forcing_backend(backend, algorithm):
    with joblib.parallel_backend(backend):
        X, y = datasets.make_classification(n_samples=30, n_features=5, n_redundant=0, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm=algorithm, n_jobs=2)
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        clf.kneighbors(X_test)
        clf.kneighbors_graph(X_test, mode='distance')