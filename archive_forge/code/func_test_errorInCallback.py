from __future__ import annotations
import unittest as pyunit
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import reporter, unittest, util
from twisted.trial.test import detests
def test_errorInCallback(self) -> None:
    result = self.runTest('test_errorInCallback')
    self.assertFalse(result.wasSuccessful())
    self.assertEqual(result.testsRun, 1)
    self.assertEqual(len(result.errors), 1)