from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def require_slack_sdk(required: bool=False):
    """
    Wrapper for `resolve_slack_sdk` that can be used as a decorator
    """

    def decorator(func):
        return require_missing_wrapper(resolver=resolve_slack_sdk, func=func, required=required)
    return decorator