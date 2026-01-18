from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_make_constraint_unknown():
    """Check that an informative error is raised when an unknown constraint is passed"""
    with pytest.raises(ValueError, match='Unknown constraint'):
        make_constraint('not a valid constraint')