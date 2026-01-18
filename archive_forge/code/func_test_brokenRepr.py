import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_brokenRepr(self):
    b = Breakable()
    b.breakRepr = True
    reflect.safe_str(b)