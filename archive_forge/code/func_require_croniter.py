from lazyops.utils import resolve_missing, require_missing_wrapper
def require_croniter(required: bool=False):
    """
    Wrapper for `resolve_croniter` that can be used as a decorator
    """

    def decorator(func):
        return require_missing_wrapper(resolver=resolve_croniter, func=func, required=required)
    return decorator