import errno
import io
import os
import sys
import socket
import select
import struct
import tempfile
import itertools
from . import reduction
from . import util
from . import AuthenticationError, BufferTooShort
from ._ext import _billiard
from .compat import setblocking, send_offset
from time import monotonic
from .reduction import ForkingPickler
def send_offset(self, buf, offset):
    return send_offset(self.fileno(), buf, offset)