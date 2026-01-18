from functools import wraps
from jedi import debug
def inference_state_method_cache(default=_NO_DEFAULT):

    def decorator(func):
        return _memoize_default(default=default)(func)
    return decorator