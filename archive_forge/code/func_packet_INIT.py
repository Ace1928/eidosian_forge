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
def packet_INIT(self, data):
    version, = struct.unpack('!L', data[:4])
    self.version = min(list(self.versions) + [version])
    data = data[4:]
    ext = {}
    while data:
        extName, data = getNS(data)
        extData, data = getNS(data)
        ext[extName] = extData
    ourExt = self.client.gotVersion(version, ext)
    ourExtData = b''
    for k, v in ourExt.items():
        ourExtData += NS(k) + NS(v)
    self.sendPacket(FXP_VERSION, struct.pack('!L', self.version) + ourExtData)