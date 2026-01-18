import random
import inspect
import backoff
import functools
import aiohttpx
from typing import Callable, List, Optional, TypeVar, TYPE_CHECKING
def build_name_func(func: Callable, *args, **kwargs) -> str:
    """
        Build the name function
        """
    nonlocal base_name
    func = inspect.unwrap(func)
    func_name = func.__qualname__ if include_classname else func.__name__
    if special_names:
        for name in special_names:
            if name in func_name.lower():
                func_name = func_name.replace(f'a{name}_', '').replace(f'{name}_', '').replace('__', '_')
                return f'{base_name}.{name}_{func_name}'
    if function_names:
        for name in function_names:
            if name in func_name.lower():
                return f'{base_name}.{func_name}'
    if include_http_methods:
        for method in {'get', 'post', 'put', 'delete'}:
            if method in func_name.lower():
                return f'{base_name}.{method}'
    return f'{base_name}.{func_name}'