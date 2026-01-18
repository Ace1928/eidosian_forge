from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
def test_failGreaterThan64k(self) -> defer.Deferred[None]:
    """
        A test which fails in the callback of the returned L{defer.Deferred}
        with a very long string.
        """
    return deferLater(reactor, 0, self.fail, 'I fail later: ' + 'x' * 2 ** 16)