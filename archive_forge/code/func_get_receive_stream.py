import asyncio
from asyncio import AbstractEventLoop, Queue
from typing import AsyncIterator, Generic, TypeVar
def get_receive_stream(self) -> _ReceiveStream[T]:
    """Get a reader for the channel."""
    return _ReceiveStream[T](queue=self._queue, done=self._done)