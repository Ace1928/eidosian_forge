from __future__ import division
import base64
import collections
import errno
import functools
import glob
import os
import re
import socket
import struct
import sys
import warnings
from collections import defaultdict
from collections import namedtuple
from . import _common
from . import _psposix
from . import _psutil_linux as cext
from . import _psutil_posix as cext_posix
from ._common import NIC_DUPLEX_FULL
from ._common import NIC_DUPLEX_HALF
from ._common import NIC_DUPLEX_UNKNOWN
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import bcat
from ._common import cat
from ._common import debug
from ._common import decode
from ._common import get_procfs_path
from ._common import isfile_strict
from ._common import memoize
from ._common import memoize_when_activated
from ._common import open_binary
from ._common import open_text
from ._common import parse_environ_block
from ._common import path_exists_strict
from ._common import supports_ipv6
from ._common import usage_percent
from ._compat import PY3
from ._compat import FileNotFoundError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import b
from ._compat import basestring
def read_procfs():
    with open_text('%s/diskstats' % get_procfs_path()) as f:
        lines = f.readlines()
    for line in lines:
        fields = line.split()
        flen = len(fields)
        if flen == 15:
            name = fields[3]
            reads = int(fields[2])
            reads_merged, rbytes, rtime, writes, writes_merged, wbytes, wtime, _, busy_time, _ = map(int, fields[4:14])
        elif flen == 14 or flen >= 18:
            name = fields[2]
            reads, reads_merged, rbytes, rtime, writes, writes_merged, wbytes, wtime, _, busy_time, _ = map(int, fields[3:14])
        elif flen == 7:
            name = fields[2]
            reads, rbytes, writes, wbytes = map(int, fields[3:])
            rtime = wtime = reads_merged = writes_merged = busy_time = 0
        else:
            raise ValueError('not sure how to interpret line %r' % line)
        yield (name, reads, writes, rbytes, wbytes, rtime, wtime, reads_merged, writes_merged, busy_time)