import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def test_inCallback(self):
    """
            Log an error in an asynchronous callback.
            """
    return task.deferLater(reactor, 0, lambda: log.err(makeFailure()))