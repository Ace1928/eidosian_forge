import asyncio
from unittest.mock import MagicMock
import pytest
import tornado
from nbclient.util import run_hook, run_sync
def test_nested_asyncio_with_tornado():
    asyncio.set_event_loop(asyncio.new_event_loop())
    ioloop = tornado.ioloop.IOLoop.current()

    async def some_async_function():
        future: asyncio.Future = asyncio.ensure_future(asyncio.sleep(0.1))
        ioloop.add_future(future, lambda f: f.result())
        await future
        return 42

    def some_sync_function():
        return run_sync(some_async_function)()

    async def run():
        assert await some_async_function() == 42
        assert some_sync_function() == 42
    ioloop.run_sync(run)