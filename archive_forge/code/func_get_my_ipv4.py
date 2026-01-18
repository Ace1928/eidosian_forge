import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def get_my_ipv4():
    """Returns the actual ipv4 of the local machine.

    This code figures out what source address would be used if some traffic
    were to be sent out to some well known address on the Internet. In this
    case, IP from RFC5737 is used, but the specific address does not
    matter much. No traffic is actually sent.

    .. versionadded:: 1.1

    .. versionchanged:: 1.2.1
       Return ``'127.0.0.1'`` if there is no default interface.
    """
    try:
        csock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        csock.connect(('192.0.2.0', 80))
        addr, port = csock.getsockname()
        csock.close()
        return addr
    except socket.error:
        return _get_my_ipv4_address()