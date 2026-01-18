import socket
import sys
import time
import random
import dns.exception
import dns.flags
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig
from ._compat import xrange, string_types
def override_system_resolver(resolver=None):
    """Override the system resolver routines in the socket module with
    versions which use dnspython's resolver.

    This can be useful in testing situations where you want to control
    the resolution behavior of python code without having to change
    the system's resolver settings (e.g. /etc/resolv.conf).

    The resolver to use may be specified; if it's not, the default
    resolver will be used.

    resolver, a ``dns.resolver.Resolver`` or ``None``, the resolver to use.
    """
    if resolver is None:
        resolver = get_default_resolver()
    global _resolver
    _resolver = resolver
    socket.getaddrinfo = _getaddrinfo
    socket.getnameinfo = _getnameinfo
    socket.getfqdn = _getfqdn
    socket.gethostbyname = _gethostbyname
    socket.gethostbyname_ex = _gethostbyname_ex
    socket.gethostbyaddr = _gethostbyaddr