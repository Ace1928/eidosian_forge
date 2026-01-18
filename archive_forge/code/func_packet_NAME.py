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
def packet_NAME(self, data):
    d, data = self._parseRequest(data)
    count, = struct.unpack('!L', data[:4])
    data = data[4:]
    files = []
    for i in range(count):
        filename, data = getNS(data)
        longname, data = getNS(data)
        attrs, data = self._parseAttributes(data)
        files.append((filename, longname, attrs))
    d.callback(files)