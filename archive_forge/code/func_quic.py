import base64
import contextlib
import enum
import errno
import os
import os.path
import selectors
import socket
import struct
import time
from typing import Any, Dict, Optional, Tuple, Union
import dns._features
import dns.exception
import dns.inet
import dns.message
import dns.name
import dns.quic
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.serial
import dns.transaction
import dns.tsig
import dns.xfr
def quic(q: dns.message.Message, where: str, timeout: Optional[float]=None, port: int=853, source: Optional[str]=None, source_port: int=0, one_rr_per_rrset: bool=False, ignore_trailing: bool=False, connection: Optional[dns.quic.SyncQuicConnection]=None, verify: Union[bool, str]=True, server_hostname: Optional[str]=None) -> dns.message.Message:
    """Return the response obtained after sending a query via DNS-over-QUIC.

    *q*, a ``dns.message.Message``, the query to send.

    *where*, a ``str``, the nameserver IP address.

    *timeout*, a ``float`` or ``None``, the number of seconds to wait before the query
    times out. If ``None``, the default, wait forever.

    *port*, a ``int``, the port to send the query to. The default is 853.

    *source*, a ``str`` containing an IPv4 or IPv6 address, specifying the source
    address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message. The default is
    0.

    *one_rr_per_rrset*, a ``bool``. If ``True``, put each RR into its own RRset.

    *ignore_trailing*, a ``bool``. If ``True``, ignore trailing junk at end of the
    received message.

    *connection*, a ``dns.quic.SyncQuicConnection``.  If provided, the
    connection to use to send the query.

    *verify*, a ``bool`` or ``str``.  If a ``True``, then TLS certificate verification
    of the server is done using the default CA bundle; if ``False``, then no
    verification is done; if a `str` then it specifies the path to a certificate file or
    directory which will be used for verification.

    *server_hostname*, a ``str`` containing the server's hostname.  The
    default is ``None``, which means that no hostname is known, and if an
    SSL context is created, hostname checking will be disabled.

    Returns a ``dns.message.Message``.
    """
    if not dns.quic.have_quic:
        raise NoDOQ('DNS-over-QUIC is not available.')
    q.id = 0
    wire = q.to_wire()
    the_connection: dns.quic.SyncQuicConnection
    the_manager: dns.quic.SyncQuicManager
    if connection:
        manager: contextlib.AbstractContextManager = contextlib.nullcontext(None)
        the_connection = connection
    else:
        manager = dns.quic.SyncQuicManager(verify_mode=verify, server_name=server_hostname)
        the_manager = manager
    with manager:
        if not connection:
            the_connection = the_manager.connect(where, port, source, source_port)
        start, expiration = _compute_times(timeout)
        with the_connection.make_stream(timeout) as stream:
            stream.send(wire, True)
            wire = stream.receive(_remaining(expiration))
        finish = time.time()
    r = dns.message.from_wire(wire, keyring=q.keyring, request_mac=q.request_mac, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing)
    r.time = max(finish - start, 0.0)
    if not q.is_response(r):
        raise BadResponse
    return r