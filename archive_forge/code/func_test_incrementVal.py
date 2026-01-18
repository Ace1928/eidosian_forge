from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_incrementVal(self):
    """
        L{MemCacheProtocol.increment} takes an optional argument C{value} which
        replaces the default value of 1 when specified.
        """
    return self._test(self.proto.increment(b'foo', 8), b'incr foo 8\r\n', b'4\r\n', 4)