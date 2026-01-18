import re
import struct
import sys
import eventlet
from eventlet import patcher
from eventlet.green import _socket_nodns
from eventlet.green import os
from eventlet.green import time
from eventlet.green import select
from eventlet.green import ssl
def is_ipv6_addr(host):
    """Return True if host is a valid IPv6 address"""
    if not isinstance(host, str):
        return False
    host = host.split('%', 1)[0]
    try:
        dns.ipv6.inet_aton(host)
    except dns.exception.SyntaxError:
        return False
    else:
        return True