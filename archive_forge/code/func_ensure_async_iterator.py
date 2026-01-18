import asyncio
import inspect
from collections import deque
from typing import (
def ensure_async_iterator(iterable: Union[Iterable, AsyncIterable]) -> AsyncIterator:
    if hasattr(iterable, '__anext__'):
        return cast(AsyncIterator, iterable)
    elif hasattr(iterable, '__aiter__'):
        return cast(AsyncIterator, iterable.__aiter__())
    else:

        class AsyncIteratorWrapper:

            def __init__(self, iterable: Iterable):
                self._iterator = iter(iterable)

            async def __anext__(self):
                try:
                    return next(self._iterator)
                except StopIteration:
                    raise StopAsyncIteration

            def __aiter__(self):
                return self
        return AsyncIteratorWrapper(iterable)