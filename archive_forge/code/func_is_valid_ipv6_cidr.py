import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def is_valid_ipv6_cidr(address):
    """Verify that address represents a valid IPv6 CIDR address.

    :param address: address to verify
    :type address: string
    :returns: true if address is valid, false otherwise

    .. versionadded:: 3.17
    """
    try:
        netaddr.IPNetwork(address, version=6).cidr
        return True
    except (TypeError, netaddr.AddrFormatError):
        return False