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
def processExited(self, reason):
    if self.delayed_call is not None:
        self.delayed_call.cancel()
    result.callback(self.data)