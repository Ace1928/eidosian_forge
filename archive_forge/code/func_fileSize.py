import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
def fileSize(file):
    """
    I'll try my damndest to determine the size of this file object.

    @param file: The file object to determine the size of.
    @type file: L{io.IOBase}

    @rtype: L{int} or L{None}
    @return: The size of the file object as an integer if it can be determined,
        otherwise return L{None}.
    """
    size = None
    if hasattr(file, 'fileno'):
        fileno = file.fileno()
        try:
            stat_ = os.fstat(fileno)
            size = stat_[stat.ST_SIZE]
        except BaseException:
            pass
        else:
            return size
    if hasattr(file, 'name') and path.exists(file.name):
        try:
            size = path.getsize(file.name)
        except BaseException:
            pass
        else:
            return size
    if hasattr(file, 'seek') and hasattr(file, 'tell'):
        try:
            try:
                file.seek(0, 2)
                size = file.tell()
            finally:
                file.seek(0, 0)
        except BaseException:
            pass
        else:
            return size
    return size