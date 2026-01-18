from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_checkAndSet(self):
    """
        L{MemCacheProtocol.checkAndSet} passes an additional cas identifier
        that the server handles to check if the data has to be updated.
        """
    return self._test(self.proto.checkAndSet(b'foo', b'bar', cas=b'1234'), b'cas foo 0 0 3 1234\r\nbar\r\n', b'STORED\r\n', True)