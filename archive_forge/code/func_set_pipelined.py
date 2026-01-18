from binascii import hexlify
from collections import deque
import socket
import threading
import time
from paramiko.common import DEBUG, io_sleep
from paramiko.file import BufferedFile
from paramiko.util import u
from paramiko.sftp import (
from paramiko.sftp_attr import SFTPAttributes
def set_pipelined(self, pipelined=True):
    """
        Turn on/off the pipelining of write operations to this file.  When
        pipelining is on, paramiko won't wait for the server response after
        each write operation.  Instead, they're collected as they come in. At
        the first non-write operation (including `.close`), all remaining
        server responses are collected.  This means that if there was an error
        with one of your later writes, an exception might be thrown from within
        `.close` instead of `.write`.

        By default, files are not pipelined.

        :param bool pipelined:
            ``True`` if pipelining should be turned on for this file; ``False``
            otherwise

        .. versionadded:: 1.5
        """
    self.pipelined = pipelined