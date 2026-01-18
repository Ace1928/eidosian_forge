import asyncio
import threading
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Awaitable, NoReturn, Optional, Union
from aioredis.exceptions import LockError, LockNotOwnedError
def reacquire(self) -> Awaitable[bool]:
    """
        Resets a TTL of an already acquired lock back to a timeout value.
        """
    if self.local.token is None:
        raise LockError('Cannot reacquire an unlocked lock')
    if self.timeout is None:
        raise LockError('Cannot reacquire a lock with no timeout')
    return self.do_reacquire()