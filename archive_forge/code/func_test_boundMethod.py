import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_boundMethod(self):
    """
        Test references search through method special attributes.
        """

    class Dummy:

        def dummy(self):
            pass
    o = Dummy()
    m = o.dummy
    self.assertIn('.__self__', reflect.objgrep(m, m.__self__, reflect.isSame))
    self.assertIn('.__self__.__class__', reflect.objgrep(m, m.__self__.__class__, reflect.isSame))
    self.assertIn('.__func__', reflect.objgrep(m, m.__func__, reflect.isSame))