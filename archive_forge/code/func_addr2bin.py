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
def addr2bin(addr):
    """Convert a string IPv4 address into an unsigned integer.

    Examples::
    >>> addr2bin('127.0.0.1')
    2130706433

    >>> addr2bin('127.0.0.1') == socket.INADDR_LOOPBACK
    1

    >>> addr2bin('255.255.255.254')
    4294967294L

    >>> addr2bin('192.168.0.1')
    3232235521L

    Unlike old DNS.addr2bin, the n, n.n, and n.n.n forms for IP addresses
    are handled as well::
    >>> addr2bin('10.65536')
    167837696
    >>> 10 * (2 ** 24) + 65536
    167837696

    >>> addr2bin('10.93.512')
    173867520
    >>> 10 * (2 ** 24) + 93 * (2 ** 16) + 512
    173867520
    """
    return struct_unpack('!L', inet_aton(addr))[0]