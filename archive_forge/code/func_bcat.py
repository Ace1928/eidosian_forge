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
def bcat(fname, fallback=_DEFAULT):
    """Same as above but opens file in binary mode."""
    return cat(fname, fallback=fallback, _open=open_binary)