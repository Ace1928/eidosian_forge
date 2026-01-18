import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_depthLimit(self):
    """
        Test the depth of references search.
        """
    a = []
    b = [a]
    c = [a, b]
    d = [a, c]
    self.assertEqual(['[0]'], reflect.objgrep(d, a, reflect.isSame, maxDepth=1))
    self.assertEqual(['[0]', '[1][0]'], reflect.objgrep(d, a, reflect.isSame, maxDepth=2))
    self.assertEqual(['[0]', '[1][0]', '[1][1][0]'], reflect.objgrep(d, a, reflect.isSame, maxDepth=3))