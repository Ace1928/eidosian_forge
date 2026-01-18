from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('integer_interval, real_interval', [(Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, -5, 5, closed='both')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, -5, 5, closed='neither')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, 4, 5, closed='both')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, 5, None, closed='left')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, 4, None, closed='neither')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, -5, 5, closed='both')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, -5, 5, closed='neither')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, 1, 2, closed='both')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, None, -5, closed='left')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, None, -4, closed='neither')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, None, 1, closed='right')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, 1, None, closed='left')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, -10, -4, closed='neither')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, -10, -4, closed='right')), (Interval(Integral, -5, 5, closed='neither'), Interval(RealNotInt, 6, 10, closed='neither')), (Interval(Integral, -5, 5, closed='neither'), Interval(RealNotInt, 6, 10, closed='left')), (Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both')), (Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both'))])
def test_generate_invalid_param_val_2_intervals(integer_interval, real_interval):
    """Check that the value generated for an interval constraint does not satisfy any of
    the interval constraints.
    """
    bad_value = generate_invalid_param_val(constraint=real_interval)
    assert not real_interval.is_satisfied_by(bad_value)
    assert not integer_interval.is_satisfied_by(bad_value)
    bad_value = generate_invalid_param_val(constraint=integer_interval)
    assert not real_interval.is_satisfied_by(bad_value)
    assert not integer_interval.is_satisfied_by(bad_value)