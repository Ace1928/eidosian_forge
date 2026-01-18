import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_brokenClassNameAttribute(self):
    """
        If a class raises an exception when accessing its C{__name__} attribute
        B{and} when calling its C{__str__} implementation, L{reflect.safe_str}
        returns 'BROKEN CLASS' instead of the class name.
        """

    class X(BTBase):
        breakName = True
    xStr = reflect.safe_str(X())
    self.assertIn('<BROKEN CLASS AT 0x', xStr)
    self.assertIn(os.path.splitext(__file__)[0], xStr)
    self.assertIn('RuntimeError: str!', xStr)