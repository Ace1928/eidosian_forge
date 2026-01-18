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
def packet_REMOVE(self, data):
    requestId = data[:4]
    data = data[4:]
    filename, data = getNS(data)
    assert data == b'', f'still have data in REMOVE: {data!r}'
    d = defer.maybeDeferred(self.client.removeFile, filename)
    d.addCallback(self._cbStatus, requestId, b'remove succeeded')
    d.addErrback(self._ebStatus, requestId, b'remove failed')