from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_invalidMultipleGetResponse(self):
    """
        If the value returned doesn't match one the expected keys of the
        current multiple C{get} command, an error is raised error in
        L{MemCacheProtocol.dataReceived}.
        """
    self.proto.getMultiple([b'foo', b'bar'])
    self.assertRaises(RuntimeError, self.proto.dataReceived, b'VALUE egg 0 7\r\nspamegg\r\nEND\r\n')