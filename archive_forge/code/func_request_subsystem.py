import os
import signal
import struct
import sys
from zope.interface import implementer
from twisted.conch.interfaces import (
from twisted.conch.ssh import channel, common, connection
from twisted.internet import interfaces, protocol
from twisted.logger import Logger
from twisted.python.compat import networkString
def request_subsystem(self, data):
    subsystem, ignored = common.getNS(data)
    log.info('Asking for subsystem "{subsystem}"', subsystem=subsystem)
    client = self.avatar.lookupSubsystem(subsystem, data)
    if client:
        pp = SSHSessionProcessProtocol(self)
        proto = wrapProcessProtocol(pp)
        client.makeConnection(proto)
        pp.makeConnection(wrapProtocol(client))
        self.client = pp
        return 1
    else:
        log.error('Failed to get subsystem')
        return 0