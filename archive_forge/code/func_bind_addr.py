import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
@bind_addr.setter
def bind_addr(self, value):
    """Set the interface on which to listen for connections."""
    if isinstance(value, tuple) and value[0] in ('', None):
        raise ValueError("Host values of '' or None are not allowed. Use '0.0.0.0' (IPv4) or '::' (IPv6) instead to listen on all active interfaces.")
    self._bind_addr = value