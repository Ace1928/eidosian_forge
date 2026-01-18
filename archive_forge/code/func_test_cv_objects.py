from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_cv_objects():
    """Check that the _CVObjects constraint accepts all current ways
    to pass cv objects."""
    constraint = _CVObjects()
    assert constraint.is_satisfied_by(5)
    assert constraint.is_satisfied_by(LeaveOneOut())
    assert constraint.is_satisfied_by([([1, 2], [3, 4]), ([3, 4], [1, 2])])
    assert constraint.is_satisfied_by(None)
    assert not constraint.is_satisfied_by('not a CV object')