import contextlib
import datetime
import functools
import importlib
import warnings
from importlib.metadata import version
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
from packaging.version import parse
from requests import HTTPError, Response
from langchain_core.pydantic_v1 import SecretStr
@contextlib.contextmanager
def mock_now(dt_value):
    """Context manager for mocking out datetime.now() in unit tests.

    Example:
    with mock_now(datetime.datetime(2011, 2, 3, 10, 11)):
        assert datetime.datetime.now() == datetime.datetime(2011, 2, 3, 10, 11)
    """

    class MockDateTime(datetime.datetime):
        """Mock datetime.datetime.now() with a fixed datetime."""

        @classmethod
        def now(cls):
            return datetime.datetime(dt_value.year, dt_value.month, dt_value.day, dt_value.hour, dt_value.minute, dt_value.second, dt_value.microsecond, dt_value.tzinfo)
    real_datetime = datetime.datetime
    datetime.datetime = MockDateTime
    try:
        yield datetime.datetime
    finally:
        datetime.datetime = real_datetime