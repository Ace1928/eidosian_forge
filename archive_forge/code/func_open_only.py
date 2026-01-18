import binascii
import os
import socket
import time
import threading
from functools import wraps
from paramiko import util
from paramiko.common import (
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
from paramiko.file import BufferedFile
from paramiko.buffered_pipe import BufferedPipe, PipeTimeout
from paramiko import pipe
from paramiko.util import ClosingContextManager
def open_only(func):
    """
    Decorator for `.Channel` methods which performs an openness check.

    :raises:
        `.SSHException` -- If the wrapped method is called on an unopened
        `.Channel`.
    """

    @wraps(func)
    def _check(self, *args, **kwds):
        if self.closed or self.eof_received or self.eof_sent or (not self.active):
            raise SSHException('Channel is not open')
        return func(self, *args, **kwds)
    return _check