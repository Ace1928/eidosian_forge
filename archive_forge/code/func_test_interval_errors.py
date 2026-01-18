from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('params, error, match', [({'type': Integral, 'left': 1.0, 'right': 2, 'closed': 'both'}, TypeError, 'Expecting left to be an int for an interval over the integers'), ({'type': Integral, 'left': 1, 'right': 2.0, 'closed': 'neither'}, TypeError, 'Expecting right to be an int for an interval over the integers'), ({'type': Integral, 'left': None, 'right': 0, 'closed': 'left'}, ValueError, "left can't be None when closed == left"), ({'type': Integral, 'left': 0, 'right': None, 'closed': 'right'}, ValueError, "right can't be None when closed == right"), ({'type': Integral, 'left': 1, 'right': -1, 'closed': 'both'}, ValueError, "right can't be less than left")])
def test_interval_errors(params, error, match):
    """Check that informative errors are raised for invalid combination of parameters"""
    with pytest.raises(error, match=match):
        Interval(**params)