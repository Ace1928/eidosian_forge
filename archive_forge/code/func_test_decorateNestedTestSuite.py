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
def test_decorateNestedTestSuite(self):
    """
        Calling L{decorate} on a test suite with nested suites will return a
        test suite that maintains the same structure, but with all tests
        decorated.
        """
    test = self.TestCase()
    suite = unittest.TestSuite([unittest.TestSuite([test])])
    decoratedTest = unittest.decorate(suite, unittest.TestDecorator)
    expected = unittest.TestSuite([unittest.TestSuite([unittest.TestDecorator(test)])])
    self.assertSuitesEqual(decoratedTest, expected)