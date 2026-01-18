import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_brokenClassAttribute(self):
    """
        If an object raises an exception when accessing its C{__class__}
        attribute, L{reflect.safe_str} uses C{type} to retrieve the class
        object.
        """
    b = NoClassAttr()
    b.breakStr = True
    bStr = reflect.safe_str(b)
    self.assertIn('NoClassAttr instance at 0x', bStr)
    self.assertIn(os.path.splitext(__file__)[0], bStr)
    self.assertIn('RuntimeError: str!', bStr)