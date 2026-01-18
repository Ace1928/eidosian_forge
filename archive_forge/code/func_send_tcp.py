from __future__ import generators
import errno
import select
import socket
import struct
import sys
import time
import dns.exception
import dns.inet
import dns.name
import dns.message
import dns.rcode
import dns.rdataclass
import dns.rdatatype
from ._compat import long, string_types, PY3
def send_tcp(sock, what, expiration=None):
    """Send a DNS message to the specified TCP socket.

    *sock*, a ``socket``.

    *what*, a ``binary`` or ``dns.message.Message``, the message to send.

    *expiration*, a ``float`` or ``None``, the absolute time at which
    a timeout exception should be raised.  If ``None``, no timeout will
    occur.

    Returns an ``(int, float)`` tuple of bytes sent and the sent time.
    """
    if isinstance(what, dns.message.Message):
        what = what.to_wire()
    l = len(what)
    tcpmsg = struct.pack('!H', l) + what
    _wait_for_writable(sock, expiration)
    sent_time = time.time()
    _net_write(sock, tcpmsg, expiration)
    return (len(tcpmsg), sent_time)