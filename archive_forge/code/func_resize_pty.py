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
def resize_pty(self, width=80, height=24, width_pixels=0, height_pixels=0):
    """
        Resize the pseudo-terminal.  This can be used to change the width and
        height of the terminal emulation created in a previous `get_pty` call.

        :param int width: new width (in characters) of the terminal screen
        :param int height: new height (in characters) of the terminal screen
        :param int width_pixels: new width (in pixels) of the terminal screen
        :param int height_pixels: new height (in pixels) of the terminal screen

        :raises:
            `.SSHException` -- if the request was rejected or the channel was
            closed
        """
    m = Message()
    m.add_byte(cMSG_CHANNEL_REQUEST)
    m.add_int(self.remote_chanid)
    m.add_string('window-change')
    m.add_boolean(False)
    m.add_int(width)
    m.add_int(height)
    m.add_int(width_pixels)
    m.add_int(height_pixels)
    self.transport._send_user_message(m)