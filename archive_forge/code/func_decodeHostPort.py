import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def decodeHostPort(line):
    """
    Decode an FTP response specifying a host and port.

    @return: a 2-tuple of (host, port).
    """
    abcdef = re.sub('[^0-9, ]', '', line)
    parsed = [int(p.strip()) for p in abcdef.split(',')]
    for x in parsed:
        if x < 0 or x > 255:
            raise ValueError('Out of range', line, x)
    a, b, c, d, e, f = parsed
    host = f'{a}.{b}.{c}.{d}'
    port = (int(e) << 8) + int(f)
    return (host, port)