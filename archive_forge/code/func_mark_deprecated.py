import inspect
import re
import types
import warnings
from functools import wraps
from typing import Any, Callable, TypeVar, Union
def mark_deprecated(func):
    """
    Mark a function as deprecated by setting a private attribute on it.
    """
    setattr(func, _DEPRECATED_MARK_ATTR_NAME, True)