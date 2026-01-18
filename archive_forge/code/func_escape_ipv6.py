import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def escape_ipv6(address):
    """Escape an IP address in square brackets if IPv6

    :param address: address to optionaly escape
    :type address: string
    :returns: string

    .. versionadded:: 3.29.0
    """
    if is_valid_ipv6(address):
        return '[%s]' % address
    return address