import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_deque(self):
    """
        Test references search through a deque object.
        """
    o = object()
    D = deque()
    D.append(None)
    D.append(o)
    self.assertIn('[1]', reflect.objgrep(D, o, reflect.isSame))