import functools
import inspect
import wrapt
from debtcollector import _utils
def removed_kwarg(old_name, message=None, version=None, removal_version=None, stacklevel=3, category=None):
    """Decorates a kwarg accepting function to deprecate a removed kwarg."""
    prefix = "Using the '%s' argument is deprecated" % old_name
    out_message = _utils.generate_message(prefix, postfix=None, message=message, version=version, removal_version=removal_version)

    @wrapt.decorator
    def wrapper(f, instance, args, kwargs):
        if old_name in kwargs:
            _utils.deprecation(out_message, stacklevel=stacklevel, category=category)
        return f(*args, **kwargs)
    return wrapper