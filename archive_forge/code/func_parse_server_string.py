import ctypes
import inspect
from pkg_resources import parse_version
import textwrap
import time
import types
import eventlet
from eventlet import tpool
import netaddr
from oslo_concurrency import lockutils
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
import six
from os_win import constants
from os_win import exceptions
def parse_server_string(server_str):
    """Parses the given server_string and returns a tuple of host and port.

    If it's not a combination of host part and port, the port element
    is an empty string. If the input is invalid expression, return a tuple of
    two empty strings.
    """
    try:
        if netaddr.valid_ipv6(server_str):
            return (server_str, '')
        if server_str.find(']:') != -1:
            address, port = server_str.replace('[', '', 1).split(']:')
            return (address, port)
        if server_str.find(':') == -1:
            return (server_str, '')
        address, port = server_str.split(':')
        return (address, port)
    except (ValueError, netaddr.AddrFormatError):
        LOG.error('Invalid server_string: %s', server_str)
        return ('', '')