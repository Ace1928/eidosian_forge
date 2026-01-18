import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def test_errorsIsolated(self):
    """
        Check that an error logged in one test doesn't fail the next test.
        """
    t1 = self.MockTest('test_single')
    t2 = self.MockTest('test_silent')
    t1(self.result)
    t2(self.result)
    self.assertEqual(len(self.result.errors), 1)
    self.assertEqual(self.result.errors[0][0], t1)
    self.assertEqual(1, self.result.successes)