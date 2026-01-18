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
def check_no_traceback(self, data):
    self.assertNotIn('Traceback', data)