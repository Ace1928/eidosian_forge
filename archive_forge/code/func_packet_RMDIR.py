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
def packet_RMDIR(self, data):
    requestId = data[:4]
    data = data[4:]
    path, data = getNS(data)
    assert data == b'', f'still have data in RMDIR: {data!r}'
    d = defer.maybeDeferred(self.client.removeDirectory, path)
    d.addCallback(self._cbStatus, requestId, b'rmdir succeeded')
    d.addErrback(self._ebStatus, requestId, b'rmdir failed')