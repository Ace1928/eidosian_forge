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
def prettyTime(seconds):
    if seconds < 60:
        return (seconds, '%d seconds' % seconds)
    if seconds < 3600:
        return (seconds, '%d minutes' % (seconds / 60))
    if seconds < 86400:
        return (seconds, '%d hours' % (seconds / 3600))
    if seconds < 604800:
        return (seconds, '%d days' % (seconds / 86400))
    else:
        return (seconds, '%d weeks' % (seconds / 604800))