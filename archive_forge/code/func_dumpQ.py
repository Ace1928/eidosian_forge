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
def dumpQ(u):
    qname, qtype, qclass = u.getQuestion()
    print('qname=%s, qtype=%d(%s), qclass=%d(%s)' % (qname, qtype, Type.typestr(qtype), qclass, Class.classstr(qclass)))