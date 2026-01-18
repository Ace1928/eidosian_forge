import types
import socket
from . import Type
from . import Class
from . import Opcode
from . import Status
import DNS
from .Base import DNSError
from struct import pack as struct_pack
from struct import unpack as struct_unpack
from socket import inet_ntoa, inet_aton, inet_ntop, AF_INET6
def getbyte(self):
    if self.offset >= len(self.buf):
        raise UnpackError('Ran off end of data')
    c = self.buf[self.offset]
    self.offset = self.offset + 1
    return c