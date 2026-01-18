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
def parse_environ_block(data):
    """Parse a C environ block of environment variables into a dictionary."""
    ret = {}
    pos = 0
    WINDOWS_ = WINDOWS
    while True:
        next_pos = data.find('\x00', pos)
        if next_pos <= pos:
            break
        equal_pos = data.find('=', pos, next_pos)
        if equal_pos > pos:
            key = data[pos:equal_pos]
            value = data[equal_pos + 1:next_pos]
            if WINDOWS_:
                key = key.upper()
            ret[key] = value
        pos = next_pos + 1
    return ret