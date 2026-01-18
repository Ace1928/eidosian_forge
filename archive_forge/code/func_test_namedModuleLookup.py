import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_namedModuleLookup(self):
    """
        L{namedModule} should return the module object for the name it is
        passed.
        """
    from twisted.python import monkey
    self.assertIs(reflect.namedModule('twisted.python.monkey'), monkey)