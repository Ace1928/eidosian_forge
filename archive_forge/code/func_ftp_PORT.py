import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def ftp_PORT(self, address):
    addr = tuple(map(int, address.split(',')))
    ip = '%d.%d.%d.%d' % tuple(addr[:4])
    port = addr[4] << 8 | addr[5]
    if self.dtpFactory is not None:
        self.cleanupDTP()
    self.dtpFactory = DTPFactory(pi=self, peerHost=self.transport.getPeer().host)
    self.dtpFactory.setTimeout(self.dtpTimeout)
    self.dtpPort = reactor.connectTCP(ip, port, self.dtpFactory)

    def connected(ignored):
        return ENTERING_PORT_MODE

    def connFailed(err):
        err.trap(PortConnectionError)
        return CANT_OPEN_DATA_CNX
    return self.dtpFactory.deferred.addCallbacks(connected, connFailed)