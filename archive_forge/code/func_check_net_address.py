from __future__ import print_function
import atexit
import contextlib
import ctypes
import errno
import functools
import gc
import inspect
import os
import platform
import random
import re
import select
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import warnings
from socket import AF_INET
from socket import AF_INET6
from socket import SOCK_STREAM
import psutil
from psutil import AIX
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import bytes2human
from psutil._common import debug
from psutil._common import memoize
from psutil._common import print_color
from psutil._common import supports_ipv6
from psutil._compat import PY3
from psutil._compat import FileExistsError
from psutil._compat import FileNotFoundError
from psutil._compat import range
from psutil._compat import super
from psutil._compat import u
from psutil._compat import unicode
from psutil._compat import which
def check_net_address(addr, family):
    """Check a net address validity. Supported families are IPv4,
    IPv6 and MAC addresses.
    """
    import ipaddress
    if enum and PY3 and (not PYPY):
        assert isinstance(family, enum.IntEnum), family
    if family == socket.AF_INET:
        octs = [int(x) for x in addr.split('.')]
        assert len(octs) == 4, addr
        for num in octs:
            assert 0 <= num <= 255, addr
        if not PY3:
            addr = unicode(addr)
        ipaddress.IPv4Address(addr)
    elif family == socket.AF_INET6:
        assert isinstance(addr, str), addr
        if not PY3:
            addr = unicode(addr)
        ipaddress.IPv6Address(addr)
    elif family == psutil.AF_LINK:
        assert re.match('([a-fA-F0-9]{2}[:|\\-]?){6}', addr) is not None, addr
    else:
        raise ValueError('unknown family %r' % family)