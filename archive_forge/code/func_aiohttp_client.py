import asyncio
import contextlib
import warnings
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional, Type, Union
import pytest
from aiohttp.helpers import isasyncgenfunction
from aiohttp.web import Application
from .test_utils import (
@pytest.fixture
def aiohttp_client(loop: asyncio.AbstractEventLoop) -> Iterator[AiohttpClient]:
    """Factory to create a TestClient instance.

    aiohttp_client(app, **kwargs)
    aiohttp_client(server, **kwargs)
    aiohttp_client(raw_server, **kwargs)
    """
    clients = []

    async def go(__param: Union[Application, BaseTestServer], *args: Any, server_kwargs: Optional[Dict[str, Any]]=None, **kwargs: Any) -> TestClient:
        if isinstance(__param, Callable) and (not isinstance(__param, (Application, BaseTestServer))):
            __param = __param(loop, *args, **kwargs)
            kwargs = {}
        else:
            assert not args, 'args should be empty'
        if isinstance(__param, Application):
            server_kwargs = server_kwargs or {}
            server = TestServer(__param, loop=loop, **server_kwargs)
            client = TestClient(server, loop=loop, **kwargs)
        elif isinstance(__param, BaseTestServer):
            client = TestClient(__param, loop=loop, **kwargs)
        else:
            raise ValueError('Unknown argument type: %r' % type(__param))
        await client.start_server()
        clients.append(client)
        return client
    yield go

    async def finalize() -> None:
        while clients:
            await clients.pop().close()
    loop.run_until_complete(finalize())