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
def get_procfs_path():
    """Return updated psutil.PROCFS_PATH constant."""
    return sys.modules['psutil'].PROCFS_PATH