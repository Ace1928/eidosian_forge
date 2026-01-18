import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def get_my_ipv6():
    """Returns the actual IPv6 address of the local machine.

    This code figures out what source address would be used if some traffic
    were to be sent out to some well known address on the Internet. In this
    case, IPv6 from RFC3849 is used, but the specific address does not
    matter much. No traffic is actually sent.

    .. versionadded:: 6.1
       Return ``'::1'`` if there is no default interface.
    """
    try:
        csock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        csock.connect(('2001:db8::1', 80))
        addr, _, _, _ = csock.getsockname()
        csock.close()
        return addr
    except socket.error:
        return _get_my_ipv6_address()