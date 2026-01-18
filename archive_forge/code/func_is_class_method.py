import inspect
def is_class_method(f):
    """Returns whether the given method is a class_method."""
    return hasattr(f, '__self__') and f.__self__ is not None