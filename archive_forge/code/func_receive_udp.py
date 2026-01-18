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
def receive_udp(sock, destination, expiration=None, ignore_unexpected=False, one_rr_per_rrset=False, keyring=None, request_mac=b'', ignore_trailing=False):
    """Read a DNS message from a UDP socket.

    *sock*, a ``socket``.

    *destination*, a destination tuple appropriate for the address family
    of the socket, specifying where the associated query was sent.

    *expiration*, a ``float`` or ``None``, the absolute time at which
    a timeout exception should be raised.  If ``None``, no timeout will
    occur.

    *ignore_unexpected*, a ``bool``.  If ``True``, ignore responses from
    unexpected sources.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, put each RR into its own
    RRset.

    *keyring*, a ``dict``, the keyring to use for TSIG.

    *request_mac*, a ``binary``, the MAC of the request (for TSIG).

    *ignore_trailing*, a ``bool``.  If ``True``, ignore trailing
    junk at end of the received message.

    Raises if the message is malformed, if network errors occur, of if
    there is a timeout.

    Returns a ``dns.message.Message`` object.
    """
    wire = b''
    while 1:
        _wait_for_readable(sock, expiration)
        wire, from_address = sock.recvfrom(65535)
        if _addresses_equal(sock.family, from_address, destination) or (dns.inet.is_multicast(destination[0]) and from_address[1:] == destination[1:]):
            break
        if not ignore_unexpected:
            raise UnexpectedSource('got a response from %s instead of %s' % (from_address, destination))
    received_time = time.time()
    r = dns.message.from_wire(wire, keyring=keyring, request_mac=request_mac, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing)
    return (r, received_time)