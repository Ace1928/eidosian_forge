import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def fill_rawq(self):
    """Fill raw queue from exactly one recv() system call.

        Block if no data is immediately available.  Set self.eof when
        connection is closed.

        """
    if self.irawq >= len(self.rawq):
        self.rawq = b''
        self.irawq = 0
    buf = self.sock.recv(50)
    self.msg('recv %r', buf)
    self.eof = not buf
    self.rawq = self.rawq + buf