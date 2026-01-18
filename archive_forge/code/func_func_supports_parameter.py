import functools
import inspect
def func_supports_parameter(func, name):
    return any((param.name == name for param in _get_callable_parameters(func)))