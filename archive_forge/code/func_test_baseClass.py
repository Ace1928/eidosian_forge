import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_baseClass(self):
    """
        If C{baseClass} is passed to L{addMethodNamesToDict}, only methods which
        are a subclass of C{baseClass} are added to the result dictionary.
        """

    class Alternate:
        pass

    class Child(Separate, Alternate):

        def good_alternate(self):
            pass
    result = {}
    addMethodNamesToDict(Child, result, 'good_', Alternate)
    self.assertEqual({'alternate': 1}, result)