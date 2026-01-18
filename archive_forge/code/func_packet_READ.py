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
def packet_READ(self, data):
    requestId = data[:4]
    data = data[4:]
    handle, data = getNS(data)
    (offset, length), data = (struct.unpack('!QL', data[:12]), data[12:])
    assert data == b'', f'still have data in READ: {data!r}'
    if handle not in self.openFiles:
        self._ebRead(failure.Failure(KeyError()), requestId)
    else:
        fileObj = self.openFiles[handle]
        d = defer.maybeDeferred(fileObj.readChunk, offset, length)
        d.addCallback(self._cbRead, requestId)
        d.addErrback(self._ebStatus, requestId, b'read failed')