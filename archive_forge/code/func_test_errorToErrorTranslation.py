from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_errorToErrorTranslation(self):
    """
        When C{D} is cancelled, and C raises a particular type of error, C{G}
        may catch that error at the point of yielding and translate it into
        a different error which may be received by application code.
        """

    def cancel(it):
        it.errback(UntranslatedError())
    a = Deferred(cancel)
    d = self.sampleInlineCB(lambda: a)
    d.cancel()
    self.assertRaises(TranslatedError, self.failureResultOf(d).raiseException)