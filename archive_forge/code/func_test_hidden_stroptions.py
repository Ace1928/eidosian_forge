from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_hidden_stroptions():
    """Check that we can have 2 StrOptions constraints, one being hidden."""

    @validate_params({'param': [StrOptions({'auto'}), Hidden(StrOptions({'warn'}))]}, prefer_skip_nested_validation=True)
    def f(param):
        pass
    f('auto')
    f('warn')
    with pytest.raises(InvalidParameterError, match="The 'param' parameter") as exc_info:
        f(param='bad')
    err_msg = str(exc_info.value)
    assert 'auto' in err_msg
    assert 'warn' not in err_msg