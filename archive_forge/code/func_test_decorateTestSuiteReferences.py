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
def test_decorateTestSuiteReferences(self):
    """
        When decorating a test suite in-place, the number of references to the
        test objects in that test suite should stay the same.

        Previously, L{unittest.decorate} recreated a test suite, so the
        original suite kept references to the test objects. This test is here
        to ensure the problem doesn't reappear again.
        """
    getrefcount = getattr(sys, 'getrefcount', None)
    if getrefcount is None:
        raise unittest.SkipTest('getrefcount not supported on this platform')
    test = self.TestCase()
    suite = unittest.TestSuite([test])
    count1 = getrefcount(test)
    unittest.decorate(suite, unittest.TestDecorator)
    count2 = getrefcount(test)
    self.assertEqual(count1, count2)