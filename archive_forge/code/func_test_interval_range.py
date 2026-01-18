from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('interval_type', [Integral, Real])
def test_interval_range(interval_type):
    """Check the range of values depending on closed."""
    interval = Interval(interval_type, -2, 2, closed='left')
    assert -2 in interval
    assert 2 not in interval
    interval = Interval(interval_type, -2, 2, closed='right')
    assert -2 not in interval
    assert 2 in interval
    interval = Interval(interval_type, -2, 2, closed='both')
    assert -2 in interval
    assert 2 in interval
    interval = Interval(interval_type, -2, 2, closed='neither')
    assert -2 not in interval
    assert 2 not in interval