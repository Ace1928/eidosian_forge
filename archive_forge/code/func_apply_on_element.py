from functools import wraps
def apply_on_element(f, args, kwargs, n):
    """
    Returns a structure with the same dimension as the specified argument,
    where each basic element is replaced by the function f applied on it. All
    other arguments stay the same.
    """
    if isinstance(n, int):
        structure = args[n]
        is_arg = True
    elif isinstance(n, str):
        structure = kwargs[n]
        is_arg = False

    def f_reduced(x):
        if hasattr(x, '__iter__'):
            return list(map(f_reduced, x))
        else:
            if is_arg:
                args[n] = x
            else:
                kwargs[n] = x
            return f(*args, **kwargs)
    return list(map(f_reduced, structure))