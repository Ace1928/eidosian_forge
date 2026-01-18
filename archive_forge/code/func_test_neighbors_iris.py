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
def test_neighbors_iris():
    for algorithm in ALGORITHMS:
        clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm=algorithm)
        clf.fit(iris.data, iris.target)
        assert_array_equal(clf.predict(iris.data), iris.target)
        clf.set_params(n_neighbors=9, algorithm=algorithm)
        clf.fit(iris.data, iris.target)
        assert np.mean(clf.predict(iris.data) == iris.target) > 0.95
        rgs = neighbors.KNeighborsRegressor(n_neighbors=5, algorithm=algorithm)
        rgs.fit(iris.data, iris.target)
        assert np.mean(rgs.predict(iris.data).round() == iris.target) > 0.95