import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_brokenUtf8(self):
    """
        Use str() for non-utf8 bytes: "b'non-utf8'"
        """
    x = b'\xff'
    xStr = reflect.safe_str(x)
    self.assertEqual(xStr, str(x))