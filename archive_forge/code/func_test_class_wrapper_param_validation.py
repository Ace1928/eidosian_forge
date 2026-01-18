from importlib import import_module
from inspect import signature
from numbers import Integral, Real
import pytest
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('func_module, class_module', PARAM_VALIDATION_CLASS_WRAPPER_LIST)
def test_class_wrapper_param_validation(func_module, class_module):
    """Check param validation for public functions that are wrappers around
    estimators.
    """
    func, func_name, func_params, required_params = _get_func_info(func_module)
    module_name, class_name = class_module.rsplit('.', 1)
    module = import_module(module_name)
    klass = getattr(module, class_name)
    parameter_constraints_func = getattr(func, '_skl_parameter_constraints')
    parameter_constraints_class = getattr(klass, '_parameter_constraints')
    parameter_constraints = {**parameter_constraints_class, **parameter_constraints_func}
    parameter_constraints = {k: v for k, v in parameter_constraints.items() if k in func_params}
    _check_function_param_validation(func, func_name, func_params, required_params, parameter_constraints)