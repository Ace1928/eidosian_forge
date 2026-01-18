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
@open_only
def set_environment_variable(self, name, value):
    """
        Set the value of an environment variable.

        .. warning::
            The server may reject this request depending on its ``AcceptEnv``
            setting; such rejections will fail silently (which is common client
            practice for this particular request type). Make sure you
            understand your server's configuration before using!

        :param str name: name of the environment variable
        :param str value: value of the environment variable

        :raises:
            `.SSHException` -- if the request was rejected or the channel was
            closed
        """
    m = Message()
    m.add_byte(cMSG_CHANNEL_REQUEST)
    m.add_int(self.remote_chanid)
    m.add_string('env')
    m.add_boolean(False)
    m.add_string(name)
    m.add_string(value)
    self.transport._send_user_message(m)