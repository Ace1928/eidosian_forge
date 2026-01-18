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
def assert_free(self, host, port=None):
    """
        Assert that the given addr is free
        in that all attempts to connect fail within the timeout
        or raise a PortNotFree exception.

        >>> free_port = find_available_local_port()

        >>> Checker().assert_free('localhost', free_port)
        >>> Checker().assert_free('127.0.0.1', free_port)
        >>> Checker().assert_free('::1', free_port)

        Also accepts an addr tuple

        >>> addr = '::1', free_port, 0, 0
        >>> Checker().assert_free(addr)

        Host might refer to a server bind address like '::', which
        should use localhost to perform the check.

        >>> Checker().assert_free('::', free_port)
        """
    if port is None and isinstance(host, abc.Sequence):
        host, port = host[:2]
    if platform.system() == 'Windows':
        host = client_host(host)
    info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    list(itertools.starmap(self._connect, info))