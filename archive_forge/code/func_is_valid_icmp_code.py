import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def is_valid_icmp_code(code):
    """Verify if ICMP code is valid.

    :param code: ICMP *code* field can be valid integer or None
    :returns: bool

    ICMP *code* field can be either None or valid integer having
    a value of 0 up to and including 255.
    """
    if code is None:
        return True
    return _is_int_in_range(code, 0, 255)