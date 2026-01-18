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
def packet_OPENDIR(self, data):
    requestId = data[:4]
    data = data[4:]
    path, data = getNS(data)
    assert data == b'', f'still have data in OPENDIR: {data!r}'
    d = defer.maybeDeferred(self.client.openDirectory, path)
    d.addCallback(self._cbOpenDirectory, requestId)
    d.addErrback(self._ebStatus, requestId, b'opendir failed')