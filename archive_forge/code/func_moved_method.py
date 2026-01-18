import functools
import inspect
import wrapt
from debtcollector import _utils
def moved_method(new_method_name, message=None, version=None, removal_version=None, stacklevel=3, category=None):
    """Decorates an *instance* method that was moved to another location."""
    if not new_method_name.endswith(_MOVED_CALLABLE_POSTFIX):
        new_method_name += _MOVED_CALLABLE_POSTFIX
    return _moved_decorator('Method', new_method_name, message=message, version=version, removal_version=removal_version, stacklevel=stacklevel, attr_postfix=_MOVED_CALLABLE_POSTFIX, category=category)