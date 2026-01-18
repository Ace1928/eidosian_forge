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
def test_deprecatedSkipWithoutReason(self):
    """
        If a test method raises L{SkipTest} with no reason, a deprecation
        warning is emitted.
        """
    self.loadSuite(self.DeprecatedReasonlessSkip)
    self.suite(self.reporter)
    warnings = self.flushWarnings([self.DeprecatedReasonlessSkip.test_1])
    self.assertEqual(1, len(warnings))
    self.assertEqual(DeprecationWarning, warnings[0]['category'])
    self.assertEqual('Do not raise unittest.SkipTest with no arguments! Give a reason for skipping tests!', warnings[0]['message'])