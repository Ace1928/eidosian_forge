import functools
import inspect
import wrapt
from debtcollector import _utils
@functools.wraps(new_func, assigned=_utils.get_assigned(new_func))
def old_new_func(*args, **kwargs):
    _utils.deprecation(out_message, stacklevel=stacklevel, category=category)
    return new_func(*args, **kwargs)