import asyncio
from asyncio import AbstractEventLoop, Queue
from typing import AsyncIterator, Generic, TypeVar
def send_nowait(self, item: T) -> None:
    """Schedule the item to be written to the queue using the original loop."""
    self._reader_loop.call_soon_threadsafe(self._queue.put_nowait, item)