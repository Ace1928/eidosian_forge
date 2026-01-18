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
def is_nonstring_iterable(obj: Any) -> bool:
    """Returns whether the obj is an iterable, but not a string

    Args:
        obj (Any): _description_

    Returns:
        bool: _description_
    """
    return isinstance(obj, Iterable) and (not isinstance(obj, str))