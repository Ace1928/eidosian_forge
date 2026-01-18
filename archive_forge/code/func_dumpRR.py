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
def dumpRR(u):
    name, type, klass, ttl, rdlength = u.getRRheader()
    typename = Type.typestr(type)
    print('name=%s, type=%d(%s), class=%d(%s), ttl=%d' % (name, type, typename, klass, Class.classstr(klass), ttl))
    mname = 'get%sdata' % typename
    if hasattr(u, mname):
        print('  formatted rdata:', getattr(u, mname)())
    else:
        print('  binary rdata:', u.getbytes(rdlength))