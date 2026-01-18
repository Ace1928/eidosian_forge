import argparse
import os
import sys
import urllib.parse  # noqa: WPS301
from importlib import import_module
from contextlib import suppress
from . import server
from . import wsgi
def parse_wsgi_bind_location(bind_addr_string):
    """Convert bind address string to a BindLocation."""
    if bind_addr_string.startswith('@'):
        return AbstractSocket(bind_addr_string[1:])
    match = urllib.parse.urlparse('//{addr}'.format(addr=bind_addr_string))
    try:
        addr = match.hostname
        port = match.port
        if addr is not None or port is not None:
            return TCPSocket(addr, port)
    except ValueError:
        pass
    return UnixSocket(path=bind_addr_string)