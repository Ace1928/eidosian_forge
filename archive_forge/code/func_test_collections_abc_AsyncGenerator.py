import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
def test_collections_abc_AsyncGenerator():
    if hasattr(collections.abc, 'AsyncGenerator'):
        assert isinstance(async_range(10), collections.abc.AsyncGenerator)