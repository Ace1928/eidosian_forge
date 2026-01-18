from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_errorAdd(self):
    """
        Test an erroneous add: if a L{MemCacheProtocol.add} is called but the
        key already exists on the server, it returns a B{NOT STORED} answer,
        which calls back the resulting L{Deferred} with C{False}.
        """
    return self._test(self.proto.add(b'foo', b'bar'), b'add foo 0 0 3\r\nbar\r\n', b'NOT STORED\r\n', False)