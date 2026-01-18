from builtins import abs as _abs
def iconcat(a, b):
    """Same as a += b, for a and b sequences."""
    if not hasattr(a, '__getitem__'):
        msg = "'%s' object can't be concatenated" % type(a).__name__
        raise TypeError(msg)
    a += b
    return a