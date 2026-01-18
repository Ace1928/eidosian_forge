import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
def test_isasyncgenfunction_is_not_inherited_by_wrappers():

    @wraps(async_range)
    def async_range_wrapper(*args, **kwargs):
        return async_range(*args, **kwargs)
    assert not isasyncgenfunction(async_range_wrapper)
    assert isasyncgenfunction(async_range_wrapper.__wrapped__)