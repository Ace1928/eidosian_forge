import fcntl
import getpass
import os
import signal
import struct
import sys
import tty
from typing import List, Tuple
from twisted.conch.client import connect, default
from twisted.conch.client.options import ConchOptions
from twisted.conch.error import ConchError
from twisted.conch.ssh import channel, common, connection, forwarding, session
from twisted.internet import reactor, stdio, task
from twisted.python import log, usage
from twisted.python.compat import ioType, networkString
def opt_localforward(self, f):
    """
        Forward local port to remote address (lport:host:port)
        """
    localPort, remoteHost, remotePort = f.split(':')
    localPort = int(localPort)
    remotePort = int(remotePort)
    self.localForwards.append((localPort, (remoteHost, remotePort)))