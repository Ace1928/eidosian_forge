from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('constraint', [_ArrayLikes(), _Callables(), _InstancesOf(list), _NoneConstraint(), _RandomStates(), _SparseMatrices(), _Booleans(), _VerboseHelper(), MissingValues(), MissingValues(numeric_only=True), StrOptions({'a', 'b', 'c'}), Options(Integral, {1, 2, 3}), Interval(Integral, None, None, closed='neither'), Interval(Integral, 0, 10, closed='neither'), Interval(Integral, 0, None, closed='neither'), Interval(Integral, None, 0, closed='neither'), Interval(Real, 0, 1, closed='neither'), Interval(Real, 0, None, closed='both'), Interval(Real, None, 0, closed='right'), HasMethods('fit'), _IterablesNotString(), _CVObjects()])
def test_generate_valid_param(constraint):
    """Check that the value generated does satisfy the constraint."""
    value = generate_valid_param(constraint)
    assert constraint.is_satisfied_by(value)