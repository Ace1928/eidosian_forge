import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
def hostile_coroutine_runner(coro):
    coro_iter = coro.__await__()
    value = None
    while True:
        try:
            if value == 'hit me':
                value = coro_iter.throw(MyTestError())
            elif value == 'number me':
                value = coro_iter.send(1)
            else:
                assert value in (None, 'next me')
                value = coro_iter.__next__()
        except StopIteration as exc:
            return exc.value