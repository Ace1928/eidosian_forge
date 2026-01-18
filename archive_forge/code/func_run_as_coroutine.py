from __future__ import unicode_literals
from .base import EventLoop, INPUT_TIMEOUT
from ..terminal.win32_input import ConsoleInputReader
from .callbacks import EventLoopCallbacks
from .asyncio_base import AsyncioTimeout
import asyncio
@asyncio.coroutine
def run_as_coroutine(self, stdin, callbacks):
    """
        The input 'event loop'.
        """
    assert isinstance(callbacks, EventLoopCallbacks)
    if self.closed:
        raise Exception('Event loop already closed.')
    timeout = AsyncioTimeout(INPUT_TIMEOUT, callbacks.input_timeout, self.loop)
    self.running = True
    try:
        while self.running:
            timeout.reset()
            try:
                g = iter(self.loop.run_in_executor(None, self._console_input_reader.read))
                while True:
                    yield next(g)
            except StopIteration as e:
                keys = e.args[0]
            for k in keys:
                callbacks.feed_key(k)
    finally:
        timeout.stop()