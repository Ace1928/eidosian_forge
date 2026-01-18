from __future__ import annotations
from io import StringIO
from twisted.trial import reporter, runner, unittest
def test_interruptInTest(self) -> None:
    runner.TrialSuite([self.suite]).run(self.reporter)
    self.assertTrue(self.reporter.shouldStop)
    self.assertEqual(2, self.reporter.testsRun)
    self.assertFalse(InterruptInTestTests.test_03_doNothing_run, 'test_03_doNothing ran.')