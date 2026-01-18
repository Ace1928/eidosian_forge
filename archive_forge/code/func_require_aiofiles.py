from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def require_aiofiles(required: bool=False):
    """
    Wrapper for `resolve_aiofiles` that can be used as a decorator
    """

    def decorator(func):
        return require_missing_wrapper(resolver=resolve_aiofiles, func=func, required=required)
    return decorator