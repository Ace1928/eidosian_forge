import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_namedAnyClassLookup(self):
    """
        L{namedAny} should return the class object for the name it is passed.
        """
    self.assertIs(reflect.namedAny('twisted.test.test_reflect.Summer'), Summer)