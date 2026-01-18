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
def packet_STATUS(self, data):
    d, data = self._parseRequest(data)
    code, = struct.unpack('!L', data[:4])
    data = data[4:]
    if len(data) >= 4:
        msg, data = getNS(data)
        if len(data) >= 4:
            lang, data = getNS(data)
        else:
            lang = b''
    else:
        msg = b''
        lang = b''
    if code == FX_OK:
        d.callback((msg, lang))
    elif code == FX_EOF:
        d.errback(EOFError(msg))
    elif code == FX_OP_UNSUPPORTED:
        d.errback(NotImplementedError(msg))
    else:
        d.errback(SFTPError(code, nativeString(msg), lang))