from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
def class_instance_overload(target):
    """
    Decorator to add an overload for target that applies when the first argument
    is a ClassInstanceType.
    """

    def decorator(func):

        @wraps(func)
        def wrapped(*args, **kwargs):
            if not isinstance(args[0], ClassInstanceType):
                return
            return func(*args, **kwargs)
        if target is not complex:
            params = list(inspect.signature(wrapped).parameters)
            assert params == _get_args(len(params))
        return overload(target)(wrapped)
    return decorator