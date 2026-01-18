import time
import socket
import argparse
import sys
import itertools
import contextlib
import platform
from collections import abc
import urllib.parse
from tempora import timing
def find_available_local_port():
    """
    Find a free port on localhost.

    >>> 0 < find_available_local_port() < 65536
    True
    """
    infos = socket.getaddrinfo(None, 0, socket.AF_UNSPEC, socket.SOCK_STREAM)
    family, proto, _, _, addr = next(iter(infos))
    sock = socket.socket(family, proto)
    sock.bind(addr)
    addr, port = sock.getsockname()[:2]
    sock.close()
    return port