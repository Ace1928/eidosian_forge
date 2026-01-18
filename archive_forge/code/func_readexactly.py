import socket
from . import coroutines
from . import events
from . import futures
from . import protocols
from .coroutines import coroutine
from .log import logger
@coroutine
def readexactly(self, n):
    if self._exception is not None:
        raise self._exception
    blocks = []
    while n > 0:
        block = (yield from self.read(n))
        if not block:
            partial = b''.join(blocks)
            raise IncompleteReadError(partial, len(partial) + n)
        blocks.append(block)
        n -= len(block)
    return b''.join(blocks)