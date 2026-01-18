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
def test_issue108(self):
    input = textwrap.dedent('            def spam():\n            u"y\\xe4y"\n            \x08\n            spam(')
    deferred = self.run_bpython(input)
    return deferred.addCallback(self.check_no_traceback)