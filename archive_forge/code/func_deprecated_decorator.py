import sys
import types
import inspect
from functools import wraps, update_wrapper
from sympy.utilities.exceptions import sympy_deprecation_warning
def deprecated_decorator(wrapped):
    if hasattr(wrapped, '__mro__'):

        class wrapper(wrapped):
            __doc__ = wrapped.__doc__
            __module__ = wrapped.__module__
            _sympy_deprecated_func = wrapped
            if '__new__' in wrapped.__dict__:

                def __new__(cls, *args, **kwargs):
                    sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
                    return super().__new__(cls, *args, **kwargs)
            else:

                def __init__(self, *args, **kwargs):
                    sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
                    super().__init__(*args, **kwargs)
        wrapper.__name__ = wrapped.__name__
    else:

        @wraps(wrapped)
        def wrapper(*args, **kwargs):
            sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
            return wrapped(*args, **kwargs)
        wrapper._sympy_deprecated_func = wrapped
    return wrapper