from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_iterable_not_string():
    """Check that a string does not satisfy the _IterableNotString constraint."""
    constraint = _IterablesNotString()
    assert constraint.is_satisfied_by([1, 2, 3])
    assert constraint.is_satisfied_by(range(10))
    assert not constraint.is_satisfied_by('some string')