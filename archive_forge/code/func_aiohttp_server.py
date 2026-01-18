import asyncio
import contextlib
import warnings
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional, Type, Union
import pytest
from aiohttp.helpers import isasyncgenfunction
from aiohttp.web import Application
from .test_utils import (
@pytest.fixture
def aiohttp_server(loop: asyncio.AbstractEventLoop) -> Iterator[AiohttpServer]:
    """Factory to create a TestServer instance, given an app.

    aiohttp_server(app, **kwargs)
    """
    servers = []

    async def go(app, *, port=None, **kwargs):
        server = TestServer(app, port=port)
        await server.start_server(loop=loop, **kwargs)
        servers.append(server)
        return server
    yield go

    async def finalize() -> None:
        while servers:
            await servers.pop().close()
    loop.run_until_complete(finalize())