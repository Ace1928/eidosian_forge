from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def require_k8s(is_sync: bool=False, is_async: bool=True, is_operator: bool=False, required: bool=True):
    """
    Wrapper for `resolve_k8s` that can be used as a decorator
    """

    def decorator(func):
        return require_missing_wrapper(resolver=resolve_k8s, func=func, is_sync=is_sync, is_async=is_async, is_operator=is_operator, required=required)
    return decorator