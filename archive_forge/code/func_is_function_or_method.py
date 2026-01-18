import inspect
def is_function_or_method(obj):
    """Check if an object is a function or method.

    Args:
        obj: The Python object in question.

    Returns:
        True if the object is an function or method.
    """
    return inspect.isfunction(obj) or inspect.ismethod(obj) or is_cython(obj)