import warnings
def deprecated_func(*args, **kwargs):
    warnings.warn(warn_msg, DeprecationWarning, stacklevel=2)
    return func(*args, **kwargs)