from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('constraint', [_ArrayLikes(), _InstancesOf(list), _Callables(), _NoneConstraint(), _RandomStates(), _SparseMatrices(), _Booleans(), Interval(Integral, None, None, closed='neither')])
def test_generate_invalid_param_val_all_valid(constraint):
    """Check that the function raises NotImplementedError when there's no invalid value
    for the constraint.
    """
    with pytest.raises(NotImplementedError):
        generate_invalid_param_val(constraint)