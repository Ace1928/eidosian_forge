from functools import wraps
from jedi import debug
def inference_state_as_method_param_cache():

    def decorator(call):
        return _memoize_default(second_arg_is_inference_state=True)(call)
    return decorator