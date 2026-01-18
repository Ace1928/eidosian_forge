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
def send_exit_status(self, status):
    """
        Send the exit status of an executed command to the client.  (This
        really only makes sense in server mode.)  Many clients expect to
        get some sort of status code back from an executed command after
        it completes.

        :param int status: the exit code of the process

        .. versionadded:: 1.2
        """
    m = Message()
    m.add_byte(cMSG_CHANNEL_REQUEST)
    m.add_int(self.remote_chanid)
    m.add_string('exit-status')
    m.add_boolean(False)
    m.add_int(status)
    self.transport._send_user_message(m)