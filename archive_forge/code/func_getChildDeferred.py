from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def getChildDeferred():
    d = Deferred()

    def _eb(result):
        childResultHolder[0] = result.check(CancelledError)
        return result
    d.addErrback(_eb)
    return d