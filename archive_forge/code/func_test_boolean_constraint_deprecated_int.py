from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_boolean_constraint_deprecated_int():
    """Check that validate_params raise a deprecation message but still passes
    validation when using an int for a parameter accepting a boolean.
    """

    @validate_params({'param': ['boolean']}, prefer_skip_nested_validation=True)
    def f(param):
        pass
    f(True)
    f(np.bool_(False))