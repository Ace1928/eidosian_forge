from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def deferMeMore(result):
    result.trap(CancelledError)
    return moreDeferred