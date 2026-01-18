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
def requestRemoteForwarding(self, remotePort, hostport):
    data = forwarding.packGlobal_tcpip_forward(('0.0.0.0', remotePort))
    d = self.sendGlobalRequest(b'tcpip-forward', data, wantReply=1)
    log.msg(f'requesting remote forwarding {remotePort}:{hostport}')
    d.addCallback(self._cbRemoteForwarding, remotePort, hostport)
    d.addErrback(self._ebRemoteForwarding, remotePort, hostport)