import warnings
def function_deprecated_by(func):
    """ Return a function that warns it is deprecated by another function.

        Returns a new function that warns it is deprecated by function
        ``func``, then acts as a pass-through wrapper for ``func``.

    """
    try:
        func_name = func.__name__
    except AttributeError:
        func_name = func.__func__.__name__
    warn_msg = 'Use %s instead' % func_name

    def deprecated_func(*args, **kwargs):
        warnings.warn(warn_msg, DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return deprecated_func