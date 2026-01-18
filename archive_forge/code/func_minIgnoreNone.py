import logging
def minIgnoreNone(a, b):
    """
    Return the min of two numbers, ignoring None
    """
    if a is None:
        return b
    if b is None:
        return a
    if a < b:
        return a
    return b