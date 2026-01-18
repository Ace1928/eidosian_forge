import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
def min_len(length):
    """
    A validator that raises `ValueError` if the initializer is called
    with a string or iterable that is shorter than *length*.

    :param int length: Minimum length of the string or iterable

    .. versionadded:: 22.1.0
    """
    return _MinLengthValidator(length)