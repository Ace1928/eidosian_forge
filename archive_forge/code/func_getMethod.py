import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def getMethod(i):
    useTestableFalse = i < 2
    if useTestableFalse:
        test = self.testableFalse
    else:
        test = self.testableTrue
    return getattr(test, methodName)