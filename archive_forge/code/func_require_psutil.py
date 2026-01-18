from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def require_psutil(required: bool=False):
    """
    Wrapper for `resolve_psutil` that can be used as a decorator
    """

    def decorator(func):
        return require_missing_wrapper(resolver=resolve_psutil, func=func, required=required)
    return decorator