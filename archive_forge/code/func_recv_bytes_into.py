import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
def recv_bytes_into(self, buf, offset=0):
    """
        Receive bytes data into a writeable bytes-like object.
        Return the number of bytes read.
        """
    self._check_closed()
    self._check_readable()
    with memoryview(buf) as m:
        itemsize = m.itemsize
        bytesize = itemsize * len(m)
        if offset < 0:
            raise ValueError('negative offset')
        elif offset > bytesize:
            raise ValueError('offset too large')
        result = self._recv_bytes()
        size = result.tell()
        if bytesize < offset + size:
            raise BufferTooShort(result.getvalue())
        result.seek(0)
        result.readinto(m[offset // itemsize:(offset + size) // itemsize])
        return size