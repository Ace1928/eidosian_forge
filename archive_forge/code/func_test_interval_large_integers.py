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
def test_interval_large_integers(interval_type):
    """Check that Interval constraint work with large integers.

    non-regression test for #26648.
    """
    interval = Interval(interval_type, 0, 2, closed='neither')
    assert 2 ** 65 not in interval
    assert 2 ** 128 not in interval
    assert float(2 ** 65) not in interval
    assert float(2 ** 128) not in interval
    interval = Interval(interval_type, 0, 2 ** 128, closed='neither')
    assert 2 ** 65 in interval
    assert 2 ** 128 not in interval
    assert float(2 ** 65) in interval
    assert float(2 ** 128) not in interval
    assert 2 ** 1024 not in interval