from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_cascadeCancellingOnCancel(self):
    """
        When C{D} cancelled, C{C} will be immediately cancelled too.
        """
    childResultHolder = ['FAILURE']

    def getChildDeferred():
        d = Deferred()

        def _eb(result):
            childResultHolder[0] = result.check(CancelledError)
            return result
        d.addErrback(_eb)
        return d
    d = self.sampleInlineCB(getChildDeferred=getChildDeferred)
    d.addErrback(lambda result: None)
    d.cancel()
    self.assertEqual(childResultHolder[0], CancelledError, 'no cascade cancelling occurs')