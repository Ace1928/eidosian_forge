from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('type, expected_type_name', [(int, 'int'), (Integral, 'int'), (Real, 'float'), (np.ndarray, 'numpy.ndarray')])
def test_instances_of_type_human_readable(type, expected_type_name):
    """Check the string representation of the _InstancesOf constraint."""
    constraint = _InstancesOf(type)
    assert str(constraint) == f"an instance of '{expected_type_name}'"