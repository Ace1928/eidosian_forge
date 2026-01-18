import functools
import inspect
import wrapt
from debtcollector import _utils
def moved_function(new_func, old_func_name, old_module_name, message=None, version=None, removal_version=None, stacklevel=3, category=None):
    """Deprecates a function that was moved to another location.

    This generates a wrapper around ``new_func`` that will emit a deprecation
    warning when called. The warning message will include the new location
    to obtain the function from.
    """
    new_func_full_name = _utils.get_callable_name(new_func)
    new_func_full_name += _MOVED_CALLABLE_POSTFIX
    old_func_full_name = '.'.join([old_module_name, old_func_name])
    old_func_full_name += _MOVED_CALLABLE_POSTFIX
    prefix = _FUNC_MOVED_PREFIX_TPL % (old_func_full_name, new_func_full_name)
    out_message = _utils.generate_message(prefix, message=message, version=version, removal_version=removal_version)

    @functools.wraps(new_func, assigned=_utils.get_assigned(new_func))
    def old_new_func(*args, **kwargs):
        _utils.deprecation(out_message, stacklevel=stacklevel, category=category)
        return new_func(*args, **kwargs)
    old_new_func.__name__ = old_func_name
    old_new_func.__module__ = old_module_name
    return old_new_func