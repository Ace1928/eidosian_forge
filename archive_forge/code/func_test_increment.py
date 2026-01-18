from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_increment(self):
    """
        Test incrementing a variable: L{MemCacheProtocol.increment} returns a
        L{Deferred} which is called back with the incremented value of the
        given key.
        """
    return self._test(self.proto.increment(b'foo'), b'incr foo 1\r\n', b'4\r\n', 4)