import sys
def list_subtract(a, b):
    """Return a list ``a`` without the elements of ``b``.

    If a particular value is in ``a`` twice and ``b`` once then the returned
    list then that value will appear once in the returned list.
    """
    a_only = list(a)
    for x in b:
        if x in a_only:
            a_only.remove(x)
    return a_only