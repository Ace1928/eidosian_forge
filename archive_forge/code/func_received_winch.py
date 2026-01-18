from __future__ import unicode_literals
from ..terminal.vt100_input import InputStream
from .asyncio_base import AsyncioTimeout
from .base import EventLoop, INPUT_TIMEOUT
from .callbacks import EventLoopCallbacks
from .posix_utils import PosixStdinReader
import asyncio
import signal
def received_winch():
    self.call_from_executor(callbacks.terminal_size_changed)