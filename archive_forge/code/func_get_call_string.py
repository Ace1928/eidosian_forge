import calendar
import datetime
import datetime as dt
import importlib
import logging
import numbers
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from redis.exceptions import ResponseError
from .exceptions import TimeoutFormatError
def get_call_string(func_name: Optional[str], args: Any, kwargs: Dict[Any, Any], max_length: Optional[int]=None) -> Optional[str]:
    """
    Returns a string representation of the call, formatted as a regular
    Python function invocation statement. If max_length is not None, truncate
    arguments with representation longer than max_length.

    Args:
        func_name (str): The funtion name
        args (Any): The function arguments
        kwargs (Dict[Any, Any]): The function kwargs
        max_length (int, optional): The max length. Defaults to None.

    Returns:
        str: A String representation of the function call.
    """
    if func_name is None:
        return None
    arg_list = [as_text(truncate_long_string(repr(arg), max_length)) for arg in args]
    list_kwargs = ['{0}={1}'.format(k, as_text(truncate_long_string(repr(v), max_length))) for k, v in kwargs.items()]
    arg_list += sorted(list_kwargs)
    args = ', '.join(arg_list)
    return '{0}({1})'.format(func_name, args)