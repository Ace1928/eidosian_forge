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
def client_host(server_host):
    """
    Return the host on which a client can connect to the given listener.

    >>> client_host('192.168.0.1')
    '192.168.0.1'
    >>> client_host('0.0.0.0')
    '127.0.0.1'
    >>> client_host('::')
    '::1'
    """
    if server_host == '0.0.0.0':
        return '127.0.0.1'
    if server_host in ('::', '::0', '::0.0.0.0'):
        return '::1'
    return server_host