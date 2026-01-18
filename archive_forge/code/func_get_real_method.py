import inspect
import types
def get_real_method(obj, name):
    """Like getattr, but with a few extra sanity checks:

    - If obj is a class, ignore everything except class methods
    - Check if obj is a proxy that claims to have all attributes
    - Catch attribute access failing with any exception
    - Check that the attribute is a callable object

    Returns the method or None.
    """
    try:
        canary = getattr(obj, '_ipython_canary_method_should_not_exist_', None)
    except Exception:
        return None
    if canary is not None:
        return None
    try:
        m = getattr(obj, name, None)
    except Exception:
        return None
    if inspect.isclass(obj) and (not isinstance(m, types.MethodType)):
        return None
    if callable(m):
        return m
    return None