import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def makeFailure():
    """
    Return a new, realistic failure.
    """
    try:
        1 / 0
    except ZeroDivisionError:
        f = failure.Failure()
    return f