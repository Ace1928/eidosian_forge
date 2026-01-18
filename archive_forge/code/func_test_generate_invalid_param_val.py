from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('constraint', [Interval(Real, None, 0, closed='left'), Interval(Real, 0, None, closed='left'), Interval(Real, None, None, closed='neither'), StrOptions({'a', 'b', 'c'}), MissingValues(), MissingValues(numeric_only=True), _VerboseHelper(), HasMethods('fit'), _IterablesNotString(), _CVObjects()])
def test_generate_invalid_param_val(constraint):
    """Check that the value generated does not satisfy the constraint"""
    bad_value = generate_invalid_param_val(constraint)
    assert not constraint.is_satisfied_by(bad_value)