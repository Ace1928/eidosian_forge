from __future__ import annotations
from io import StringIO
from twisted.trial import reporter, runner, unittest
def test_interruptInSetUp(self) -> None:
    runner.TrialSuite([self.suite]).run(self.reporter)
    self.assertTrue(self.reporter.shouldStop)
    self.assertEqual(2, self.reporter.testsRun)
    self.assertFalse(InterruptInSetUpTests.test_02_run, 'test_02 ran')