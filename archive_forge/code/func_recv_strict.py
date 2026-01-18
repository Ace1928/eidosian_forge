import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
def recv_strict(self, bufsize):
    shortage = bufsize - sum((len(x) for x in self.recv_buffer))
    while shortage > 0:
        bytes_ = self.recv(min(16384, shortage))
        self.recv_buffer.append(bytes_)
        shortage -= len(bytes_)
    unified = six.b('').join(self.recv_buffer)
    if shortage == 0:
        self.recv_buffer = []
        return unified
    else:
        self.recv_buffer = [unified[bufsize:]]
        return unified[:bufsize]