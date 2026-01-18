from importlib import import_module
from inspect import signature
from numbers import Integral, Real
import pytest
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('func_module', PARAM_VALIDATION_FUNCTION_LIST)
def test_function_param_validation(func_module):
    """Check param validation for public functions that are not wrappers around
    estimators.
    """
    func, func_name, func_params, required_params = _get_func_info(func_module)
    parameter_constraints = getattr(func, '_skl_parameter_constraints')
    _check_function_param_validation(func, func_name, func_params, required_params, parameter_constraints)