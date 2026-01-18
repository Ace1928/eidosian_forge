from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def deferNext(self):
    """
        Defer the next result from my worker iterator.
        """
    self._doDeferNext = True