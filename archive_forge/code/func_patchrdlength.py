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
def patchrdlength(self):
    rdlength = unpack16bit(self.buf[self.rdstart - 2:self.rdstart])
    if rdlength == len(self.buf) - self.rdstart:
        return
    rdata = self.buf[self.rdstart:]
    save_buf = self.buf
    ok = 0
    try:
        self.buf = self.buf[:self.rdstart - 2]
        self.add16bit(len(rdata))
        self.buf = self.buf + rdata
        ok = 1
    finally:
        if not ok:
            self.buf = save_buf