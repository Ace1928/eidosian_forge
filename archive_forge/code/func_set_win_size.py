import fcntl
import os
import pty
import struct
import sys
import termios
import textwrap
import unittest
from bpython.test import TEST_CONFIG
from bpython.config import getpreferredencoding
def set_win_size(fd, rows, columns):
    s = struct.pack('HHHH', rows, columns, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, s)