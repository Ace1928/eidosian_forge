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
def complete_handshake(self):
    """
        Tells `Packetizer` that the handshake has completed.
        """
    if self.__timer:
        self.__timer.cancel()
        self.__timer_expired = False
        self.__handshake_complete = True