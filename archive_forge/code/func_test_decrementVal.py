from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_decrementVal(self):
    """
        L{MemCacheProtocol.decrement} takes an optional argument C{value} which
        replaces the default value of 1 when specified.
        """
    return self._test(self.proto.decrement(b'foo', 3), b'decr foo 3\r\n', b'5\r\n', 5)