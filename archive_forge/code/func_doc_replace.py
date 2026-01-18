import sys
def doc_replace(obj, oldval, newval):
    """Decorator to take the docstring from obj, with oldval replaced by newval

    Equivalent to ``func.__doc__ = obj.__doc__.replace(oldval, newval)``

    Parameters
    ----------
    obj : object
        The object to take the docstring from.
    oldval : string
        The string to replace from the original docstring.
    newval : string
        The string to replace ``oldval`` with.
    """
    doc = (obj.__doc__ or '').replace(oldval, newval)

    def inner(func):
        func.__doc__ = doc
        return func
    return inner