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
def exit_status_ready(self):
    """
        Return true if the remote process has exited and returned an exit
        status. You may use this to poll the process status if you don't
        want to block in `recv_exit_status`. Note that the server may not
        return an exit status in some cases (like bad servers).

        :return:
            ``True`` if `recv_exit_status` will return immediately, else
            ``False``.

        .. versionadded:: 1.7.3
        """
    return self.closed or self.status_event.is_set()