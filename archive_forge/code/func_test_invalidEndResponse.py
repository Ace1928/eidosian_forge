from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_invalidEndResponse(self):
    """
        If an END is received in response to an operation that isn't C{get},
        C{gets}, or C{stats}, an error is raised in
        L{MemCacheProtocol.dataReceived}.
        """
    self.proto.set(b'key', b'value')
    self.assertRaises(RuntimeError, self.proto.dataReceived, b'END\r\n')