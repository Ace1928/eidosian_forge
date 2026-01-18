import logging
import ssl
import sys
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, Struct
def socketwrap(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0):
    sockobj = realsocket(family, type, proto)
    if type == socket.SOCK_STREAM:
        sockobj.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sockobj