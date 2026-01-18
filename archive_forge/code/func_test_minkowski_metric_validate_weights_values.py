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
@pytest.mark.parametrize('w, err_type, err_msg', [(np.array([1, 1.5, -13]), ValueError, 'w cannot contain negative weights'), (np.array([1, 1.5, np.nan]), ValueError, 'w contains NaN'), *[(csr_container([[1, 1.5, 1]]), TypeError, 'Sparse data was passed for w, but dense data is required') for csr_container in CSR_CONTAINERS], (np.array(['a', 'b', 'c']), ValueError, 'could not convert string to float'), (np.array([]), ValueError, 'a minimum of 1 is required')])
def test_minkowski_metric_validate_weights_values(w, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        DistanceMetric.get_metric('minkowski', p=3, w=w)