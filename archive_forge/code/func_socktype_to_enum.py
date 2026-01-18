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
def socktype_to_enum(num):
    """Convert a numeric socket type value to an IntEnum member.
    If it's not a known member, return the numeric value itself.
    """
    if enum is None:
        return num
    else:
        try:
            return socket.SocketKind(num)
        except ValueError:
            return num