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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_is_sorted_by_data(csr_container):
    X = csr_container(np.arange(10).reshape(1, 10))
    assert _is_sorted_by_data(X)
    X[0, 2] = 5
    assert not _is_sorted_by_data(X)
    X = csr_container([[0, 1, 2], [3, 0, 0], [3, 4, 0], [1, 0, 2]])
    assert _is_sorted_by_data(X)
    data, indices, indptr = ([0, 4, 2, 2], [0, 1, 1, 1], [0, 2, 2, 4])
    X = csr_container((data, indices, indptr), shape=(3, 3))
    assert _is_sorted_by_data(X)