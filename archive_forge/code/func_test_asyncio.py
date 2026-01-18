import os
import sys
import pytest
from .. import (
def test_asyncio():
    import asyncio
    with pytest.raises(AsyncLibraryNotFoundError):
        current_async_library()
    ran = []

    async def this_is_asyncio():
        assert current_async_library() == 'asyncio'
        assert current_async_library() == 'asyncio'
        ran.append(True)
    asyncio.run(this_is_asyncio())
    assert ran == [True]
    with pytest.raises(AsyncLibraryNotFoundError):
        current_async_library()