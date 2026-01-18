import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def good_method(self):
    """
        A no-op method which a matching prefix to be discovered.
        """