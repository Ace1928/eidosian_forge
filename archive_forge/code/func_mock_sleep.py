import pytest
from functools import wraps, partial
import inspect
import types
@types.coroutine
def mock_sleep():
    yield 'mock_sleep'