from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_errbackCancelledErrorOnCancel(self):
    """
        When C{D} cancelled, CancelledError from C{C} will be errbacked
        through C{D}.
        """
    d = self.sampleInlineCB()
    d.cancel()
    self.assertRaises(CancelledError, self.failureResultOf(d).raiseException)