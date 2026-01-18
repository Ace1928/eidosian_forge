import warnings
import functools
@functools.wraps(obj)
def new_obj(*args, **kwargs):
    warnings.warn(message, AltairDeprecationWarning, stacklevel=1)
    return obj(*args, **kwargs)