import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def testRunner(self):
    runner = unittest.TextTestRunner(stream=io.StringIO())
    result = runner.run(unittest.TestSuite())
    self.assertIn(result, unittest.signals._results)