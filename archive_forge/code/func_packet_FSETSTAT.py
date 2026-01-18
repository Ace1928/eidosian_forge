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
def packet_FSETSTAT(self, data):
    requestId = data[:4]
    data = data[4:]
    handle, data = getNS(data)
    attrs, data = self._parseAttributes(data)
    assert data == b'', f'still have data in FSETSTAT: {data!r}'
    if handle not in self.openFiles:
        self._ebStatus(failure.Failure(KeyError()), requestId)
    else:
        fileObj = self.openFiles[handle]
        d = defer.maybeDeferred(fileObj.setAttrs, attrs)
        d.addCallback(self._cbStatus, requestId, b'fsetstat succeeded')
        d.addErrback(self._ebStatus, requestId, b'fsetstat failed')