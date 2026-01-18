import gc
import sys
import unittest as pyunit
import weakref
from io import StringIO
from twisted.internet import defer, reactor
from twisted.python.compat import _PYPY
from twisted.python.reflect import namedAny
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import (
from twisted.trial.test import erroneous
from twisted.trial.test.test_suppression import SuppressionMixin
def test_noReference(self):
    """
        Test that no reference is kept on a successful test.
        """
    test = self.__class__('test_successful')
    ref = weakref.ref(test)
    test.run(self.result)
    self.assertSuccessful(test, self.result)
    del test
    gc.collect()
    self.assertIdentical(ref(), None)