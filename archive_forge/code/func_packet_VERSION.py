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
def packet_VERSION(self, data):
    version, = struct.unpack('!L', data[:4])
    data = data[4:]
    d = {}
    while data:
        k, data = getNS(data)
        v, data = getNS(data)
        d[k] = v
    self.version = version
    self.gotServerVersion(version, d)