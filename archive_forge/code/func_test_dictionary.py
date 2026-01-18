import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_dictionary(self):
    """
        Test references search through a dictionary, as a key or as a value.
        """
    o = object()
    d1 = {None: o}
    d2 = {o: None}
    self.assertIn('[None]', reflect.objgrep(d1, o, reflect.isSame))
    self.assertIn('{None}', reflect.objgrep(d2, o, reflect.isSame))