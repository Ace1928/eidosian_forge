from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_asynchronousCancellation(self):
    """
        When C{D} is cancelled, it won't reach the callbacks added to it by
        application code until C{C} reaches the point in its callback chain
        where C{G} awaits it.  Otherwise, application code won't be able to
        track resource usage that C{D} may be using.
        """
    moreDeferred = Deferred()

    def deferMeMore(result):
        result.trap(CancelledError)
        return moreDeferred

    def deferMe():
        d = Deferred()
        d.addErrback(deferMeMore)
        return d
    d = self.sampleInlineCB(getChildDeferred=deferMe)
    d.cancel()
    self.assertNoResult(d)
    moreDeferred.callback(6543)
    self.assertEqual(self.successResultOf(d), 6544)