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
def getRRheader(self):
    name = self.getname()
    rrtype = self.get16bit()
    klass = self.get16bit()
    ttl = self.get32bit()
    rdlength = self.get16bit()
    self.rdlength = rdlength
    self.rdend = self.offset + rdlength
    return (name, rrtype, klass, ttl, rdlength)