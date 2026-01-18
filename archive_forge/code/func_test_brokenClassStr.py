import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_brokenClassStr(self):

    class X(BTBase):
        breakStr = True
    reflect.safe_str(X)
    reflect.safe_str(X())