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
def sendall_stderr(self, s):
    """
        Send data to the channel's "stderr" stream, without allowing partial
        results.  Unlike `send_stderr`, this method continues to send data
        from the given bytestring until all data has been sent or an error
        occurs. Nothing is returned.

        :param bytes s: data to send to the client as "stderr" output.

        :raises socket.timeout:
            if sending stalled for longer than the timeout set by `settimeout`.
        :raises socket.error:
            if an error occurred before the entire string was sent.

        .. versionadded:: 1.1
        """
    while s:
        sent = self.send_stderr(s)
        s = s[sent:]
    return None