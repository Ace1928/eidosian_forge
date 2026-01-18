import functools
import inspect
import warnings
@functools.wraps(func2)
def new_func2(*args, **kwargs):
    warn_deprecation(fmt2)
    return func2(*args, **kwargs)