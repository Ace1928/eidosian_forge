from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('constraint_declaration, value', [(Interval(Real, 0, 1, closed='both'), 0.42), (Interval(Integral, 0, None, closed='neither'), 42), (StrOptions({'a', 'b', 'c'}), 'b'), (Options(type, {np.float32, np.float64}), np.float64), (callable, lambda x: x + 1), (None, None), ('array-like', [[1, 2], [3, 4]]), ('array-like', np.array([[1, 2], [3, 4]])), ('sparse matrix', csr_matrix([[1, 2], [3, 4]])), ('random_state', 0), ('random_state', np.random.RandomState(0)), ('random_state', None), (_Class, _Class()), (int, 1), (Real, 0.5), ('boolean', False), ('verbose', 1), ('nan', np.nan), (MissingValues(), -1), (MissingValues(), -1.0), (MissingValues(), 2 ** 1028), (MissingValues(), None), (MissingValues(), float('nan')), (MissingValues(), np.nan), (MissingValues(), 'missing'), (HasMethods('fit'), _Estimator(a=0)), ('cv_object', 5)])
def test_is_satisfied_by(constraint_declaration, value):
    """Sanity check for the is_satisfied_by method"""
    constraint = make_constraint(constraint_declaration)
    assert constraint.is_satisfied_by(value)