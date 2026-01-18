from __future__ import division
from __future__ import print_function
import collections
import contextlib
import errno
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
@memoize
def term_supports_colors(file=sys.stdout):
    if os.name == 'nt':
        return True
    try:
        import curses
        assert file.isatty()
        curses.setupterm()
        assert curses.tigetnum('colors') > 0
    except Exception:
        return False
    else:
        return True