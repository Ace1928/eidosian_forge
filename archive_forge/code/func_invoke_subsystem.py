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
def invoke_subsystem(self, subsystem):
    """
        Request a subsystem on the server (for example, ``sftp``).  If the
        server allows it, the channel will then be directly connected to the
        requested subsystem.

        When the subsystem finishes, the channel will be closed and can't be
        reused.

        :param str subsystem: name of the subsystem being requested.

        :raises:
            `.SSHException` -- if the request was rejected or the channel was
            closed
        """
    m = Message()
    m.add_byte(cMSG_CHANNEL_REQUEST)
    m.add_int(self.remote_chanid)
    m.add_string('subsystem')
    m.add_boolean(True)
    m.add_string(subsystem)
    self._event_pending()
    self.transport._send_user_message(m)
    self._wait_for_event()