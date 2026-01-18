import copy
import itertools
import pickle
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import DistanceMetric
from sklearn.metrics._dist_metrics import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, create_memmap_backed_data
from sklearn.utils.fixes import CSR_CONTAINERS, parse_version, sp_version
def test_minkowski_metric_validate_bad_p_parameter():
    msg = 'p must be greater than 0'
    with pytest.raises(ValueError, match=msg):
        DistanceMetric.get_metric('minkowski', p=0)