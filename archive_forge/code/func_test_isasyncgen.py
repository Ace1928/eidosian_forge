import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
def test_isasyncgen():
    assert not isasyncgen(async_range)
    assert isasyncgen(async_range(10))
    if sys.version_info >= (3, 6):
        assert not isasyncgen(native_async_range)
        assert isasyncgen(native_async_range(10))