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
def cancelRemoteForwarding(self, remotePort):
    data = forwarding.packGlobal_tcpip_forward(('0.0.0.0', remotePort))
    self.sendGlobalRequest(b'cancel-tcpip-forward', data)
    log.msg(f'cancelling remote forwarding {remotePort}')
    try:
        del self.remoteForwards[remotePort]
    except Exception:
        pass
    log.msg(repr(self.remoteForwards))