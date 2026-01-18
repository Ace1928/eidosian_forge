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
def packet_SETSTAT(self, data):
    requestId = data[:4]
    data = data[4:]
    path, data = getNS(data)
    attrs, data = self._parseAttributes(data)
    if data != b'':
        self._log.warn('Still have data in SETSTAT: {data!r}', data=data)
    d = defer.maybeDeferred(self.client.setAttrs, path, attrs)
    d.addCallback(self._cbStatus, requestId, b'setstat succeeded')
    d.addErrback(self._ebStatus, requestId, b'setstat failed')