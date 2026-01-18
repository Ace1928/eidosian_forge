from __future__ import annotations
import unittest as pyunit
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import reporter, unittest, util
from twisted.trial.test import detests
def test_classTimeout(self) -> None:
    loader = pyunit.TestLoader()
    suite = loader.loadTestsFromTestCase(detests.TestClassTimeoutAttribute)
    result = reporter.TestResult()
    suite.run(result)
    self.assertEqual(len(result.errors), 1)
    assert isinstance(result.errors[0][1], Failure)
    self._wasTimeout(result.errors[0][1])