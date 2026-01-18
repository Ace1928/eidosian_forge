from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_hidden_constraint():
    """Check that internal constraints are not exposed in the error message."""

    @validate_params({'param': [Hidden(list), dict]}, prefer_skip_nested_validation=True)
    def f(param):
        pass
    f({'a': 1, 'b': 2, 'c': 3})
    f([1, 2, 3])
    with pytest.raises(InvalidParameterError, match="The 'param' parameter") as exc_info:
        f(param='bad')
    err_msg = str(exc_info.value)
    assert "an instance of 'dict'" in err_msg
    assert "an instance of 'list'" not in err_msg