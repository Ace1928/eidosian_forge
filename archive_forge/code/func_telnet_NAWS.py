import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def telnet_NAWS(self, data):
    if len(data) == 4:
        width, height = struct.unpack('!HH', b''.join(data))
        self.protocol.terminalProtocol.terminalSize(width, height)
    else:
        self._log.error('Wrong number of NAWS bytes: {nbytes}', nbytes=len(data))