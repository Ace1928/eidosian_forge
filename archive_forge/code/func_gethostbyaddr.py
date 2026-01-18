import asyncio
import functools
import pycares
import socket
import sys
from typing import (
from . import error
def gethostbyaddr(self, name: str) -> asyncio.Future:
    fut = asyncio.Future(loop=self.loop)
    cb = functools.partial(self._callback, fut)
    self._channel.gethostbyaddr(name, cb)
    return fut