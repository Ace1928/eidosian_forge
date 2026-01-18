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
@pytest.mark.filterwarnings('ignore:.*@coroutine.*:DeprecationWarning')
def test_coroutine_or_error() -> None:

    class Deferred:
        """Just kidding"""
    with ignore_coroutine_never_awaited_warnings():

        async def f() -> None:
            pass
        with pytest.raises(TypeError) as excinfo:
            coroutine_or_error(f())
        assert 'expecting an async function' in str(excinfo.value)
        import asyncio
        if sys.version_info < (3, 11):

            @asyncio.coroutine
            def generator_based_coro() -> Any:
                yield from asyncio.sleep(1)
            with pytest.raises(TypeError) as excinfo:
                coroutine_or_error(generator_based_coro())
            assert 'asyncio' in str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            coroutine_or_error(create_asyncio_future_in_new_loop())
        assert 'asyncio' in str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            coroutine_or_error(create_asyncio_future_in_new_loop)
        assert 'asyncio' in str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            coroutine_or_error(Deferred())
        assert 'twisted' in str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            coroutine_or_error(lambda: Deferred())
        assert 'twisted' in str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            coroutine_or_error(len, [[1, 2, 3]])
        assert 'appears to be synchronous' in str(excinfo.value)

        async def async_gen(_: object) -> Any:
            yield
        with pytest.raises(TypeError) as excinfo:
            coroutine_or_error(async_gen, [0])
        msg = 'expected an async function but got an async generator'
        assert msg in str(excinfo.value)
        del excinfo