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
def cache_deactivate(proc):
    """Deactivate and clear cache."""
    try:
        del proc._cache
    except AttributeError:
        pass