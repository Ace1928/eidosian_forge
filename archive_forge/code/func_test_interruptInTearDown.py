from __future__ import annotations
from io import StringIO
from twisted.trial import reporter, runner, unittest
def test_interruptInTearDown(self) -> None:
    runner.TrialSuite([self.suite]).run(self.reporter)
    self.assertEqual(1, self.reporter.testsRun)
    self.assertTrue(self.reporter.shouldStop)
    self.assertFalse(InterruptInTearDownTests.test_02_run, 'test_02 ran')