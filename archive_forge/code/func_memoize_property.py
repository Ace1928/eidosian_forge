import sys
import types
import inspect
from functools import wraps, update_wrapper
from sympy.utilities.exceptions import sympy_deprecation_warning
def memoize_property(propfunc):
    """Property decorator that caches the value of potentially expensive
    `propfunc` after the first evaluation. The cached value is stored in
    the corresponding property name with an attached underscore."""
    attrname = '_' + propfunc.__name__
    sentinel = object()

    @wraps(propfunc)
    def accessor(self):
        val = getattr(self, attrname, sentinel)
        if val is sentinel:
            val = propfunc(self)
            setattr(self, attrname, val)
        return val
    return property(accessor)