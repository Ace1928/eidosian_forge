import signal
import sys
import types
from typing import Any, TypeVar
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
from .. import _core
from .._core._tests.tutil import (
from .._util import (
from ..testing import wait_all_tasks_blocked
def test_generic_function() -> None:

    @generic_function
    def test_func(arg: T) -> T:
        """Look, a docstring!"""
        return arg
    assert test_func is test_func[int] is test_func[int, str]
    assert test_func(42) == test_func[int](42) == 42
    assert test_func.__doc__ == 'Look, a docstring!'
    assert test_func.__qualname__ == 'test_generic_function.<locals>.test_func'
    assert test_func.__name__ == 'test_func'
    assert test_func.__module__ == __name__