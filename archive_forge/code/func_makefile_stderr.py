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
def makefile_stderr(self, *params):
    """
        Return a file-like object associated with this channel's stderr
        stream.   Only channels using `exec_command` or `invoke_shell`
        without a pty will ever have data on the stderr stream.

        The optional ``mode`` and ``bufsize`` arguments are interpreted the
        same way as by the built-in ``file()`` function in Python.  For a
        client, it only makes sense to open this file for reading.  For a
        server, it only makes sense to open this file for writing.

        :returns:
            `.ChannelStderrFile` object which can be used for Python file I/O.

        .. versionadded:: 1.1
        """
    return ChannelStderrFile(*[self] + list(params))