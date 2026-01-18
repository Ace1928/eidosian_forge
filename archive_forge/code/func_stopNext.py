from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def stopNext(self):
    """
        Make the next result from my worker iterator be completion (raising
        StopIteration).
        """
    self._doStopNext = True