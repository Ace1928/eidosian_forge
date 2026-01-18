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
def test_iterateIsLeftToRightDepthFirst(self):
    """
        L{_iterateTests} returns tests in left-to-right, depth-first order.
        """
    test = self.TestCase()
    suite = runner.TestSuite([runner.TestSuite([test]), self])
    self.assertEqual([test, self], list(_iterateTests(suite)))