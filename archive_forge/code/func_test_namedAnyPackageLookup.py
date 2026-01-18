import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_namedAnyPackageLookup(self):
    """
        L{namedAny} should return the package object for the name it is passed.
        """
    import twisted.python
    self.assertIs(reflect.namedAny('twisted.python'), twisted.python)