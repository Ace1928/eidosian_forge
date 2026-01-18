from __future__ import annotations
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import requests.exceptions
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions
def if_exception_type_predicate(exception: Exception) -> bool:
    """Bound predicate for checking an exception type."""
    return isinstance(exception, exception_types)