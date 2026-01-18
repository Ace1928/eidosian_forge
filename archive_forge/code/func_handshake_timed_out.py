import errno
import os
import socket
import struct
import threading
import time
from hmac import HMAC
from paramiko import util
from paramiko.common import (
from paramiko.util import u
from paramiko.ssh_exception import SSHException, ProxyCommandFailure
from paramiko.message import Message
def handshake_timed_out(self):
    """
        Checks if the handshake has timed out.

        If `start_handshake` wasn't called before the call to this function,
        the return value will always be `False`. If the handshake completed
        before a timeout was reached, the return value will be `False`

        :return: handshake time out status, as a `bool`
        """
    if not self.__timer:
        return False
    if self.__handshake_complete:
        return False
    return self.__timer_expired