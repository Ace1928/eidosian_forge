def require_dateparser(required: bool=False):
    """
    Wrapper for `resolve_dateparser` that can be used as a decorator
    """

    def decorator(func):
        from lazyops.utils.imports import require_missing_wrapper
        return require_missing_wrapper(resolver=resolve_dateparser, func=func, required=required)
    return decorator