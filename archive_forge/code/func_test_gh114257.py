import unittest
import unittest.mock
import os.path
import sys
import test.support
from test.support import os_helper
from ctypes import *
from ctypes.util import find_library
def test_gh114257(self):
    self.assertIsNone(find_library('libc'))