import unittest
import unittest.mock
import os.path
import sys
import test.support
from test.support import os_helper
from ctypes import *
from ctypes.util import find_library
def test_glu(self):
    if self.glu is None:
        self.skipTest('lib_glu not available')
    self.glu.gluBeginCurve