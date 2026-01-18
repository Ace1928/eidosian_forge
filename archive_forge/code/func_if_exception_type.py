from __future__ import annotations
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import requests.exceptions
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions
def if_exception_type(*exception_types: type[Exception]) -> Callable[[Exception], bool]:
    """Creates a predicate to check if the exception is of a given type.

    Args:
        exception_types (Sequence[:func:`type`]): The exception types to check
            for.

    Returns:
        Callable[Exception]: A predicate that returns True if the provided
            exception is of the given type(s).
    """

    def if_exception_type_predicate(exception: Exception) -> bool:
        """Bound predicate for checking an exception type."""
        return isinstance(exception, exception_types)
    return if_exception_type_predicate