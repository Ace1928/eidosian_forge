from functools import wraps
def iter_copy(structure):
    """
    Returns a copy of an iterable object (also copying all embedded iterables).
    """
    return [iter_copy(i) if hasattr(i, '__iter__') else i for i in structure]