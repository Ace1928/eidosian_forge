import json
import inspect
from lazyops.utils.helpers import is_coro_func
from typing import Callable, Optional, List
def create_function_wrapper(function: Callable):
    """
    Creates a function wrapper as a decorator for fastapi
    """

    def inner_wrapper(handler: Callable):
        """
        The inner wrapper
        """

        async def wrapper(*args, **kwargs):
            if is_coro_func(function):
                await function(*args, **kwargs)
            else:
                function(*args, **kwargs)
            return await handler(*args, **kwargs)
        wrapper.__signature__ = inspect.Signature(parameters=[*inspect.signature(handler).parameters.values(), *filter(lambda p: p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD), inspect.signature(wrapper).parameters.values())], return_annotation=inspect.signature(handler).return_annotation)
        return wrapper
    return inner_wrapper