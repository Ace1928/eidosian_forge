import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
def packet_SYMLINK(self, data):
    requestId = data[:4]
    data = data[4:]
    linkPath, data = getNS(data)
    targetPath, data = getNS(data)
    d = defer.maybeDeferred(self.client.makeLink, linkPath, targetPath)
    d.addCallback(self._cbStatus, requestId, b'symlink succeeded')
    d.addErrback(self._ebStatus, requestId, b'symlink failed')