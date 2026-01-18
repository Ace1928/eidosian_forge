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
def cache_activate(proc):
    """Activate cache. Expects a Process instance. Cache will be
        stored as a "_cache" instance attribute.
        """
    proc._cache = {}