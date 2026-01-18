import asyncio
import os
import multiprocessing
import signal
import subprocess
import sys
import time
from binascii import hexlify
from tornado.concurrent import (
from tornado import ioloop
from tornado.iostream import PipeIOStream
from tornado.log import gen_log
import typing
from typing import Optional, Any, Callable
def set_exit_callback(self, callback: Callable[[int], None]) -> None:
    """Runs ``callback`` when this process exits.

        The callback takes one argument, the return code of the process.

        This method uses a ``SIGCHLD`` handler, which is a global setting
        and may conflict if you have other libraries trying to handle the
        same signal.  If you are using more than one ``IOLoop`` it may
        be necessary to call `Subprocess.initialize` first to designate
        one ``IOLoop`` to run the signal handlers.

        In many cases a close callback on the stdout or stderr streams
        can be used as an alternative to an exit callback if the
        signal handler is causing a problem.

        Availability: Unix
        """
    self._exit_callback = callback
    Subprocess.initialize()
    Subprocess._waiting[self.pid] = self
    Subprocess._try_cleanup_process(self.pid)