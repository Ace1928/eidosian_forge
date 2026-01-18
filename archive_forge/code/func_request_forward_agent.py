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
def request_forward_agent(self, handler):
    """
        Request for a forward SSH Agent on this channel.
        This is only valid for an ssh-agent from OpenSSH !!!

        :param handler:
            a required callable handler to use for incoming SSH Agent
            connections

        :return: True if we are ok, else False
            (at that time we always return ok)

        :raises: SSHException in case of channel problem.
        """
    m = Message()
    m.add_byte(cMSG_CHANNEL_REQUEST)
    m.add_int(self.remote_chanid)
    m.add_string('auth-agent-req@openssh.com')
    m.add_boolean(False)
    self.transport._send_user_message(m)
    self.transport._set_forward_agent_handler(handler)
    return True