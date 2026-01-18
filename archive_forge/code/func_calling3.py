from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
@inlineCallbacks
def calling3():
    yield erroring()