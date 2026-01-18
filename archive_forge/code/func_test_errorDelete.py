from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_errorDelete(self):
    """
        Test an error during a delete: if key doesn't exist on the server, it
        returns a B{NOT FOUND} answer which calls back the resulting
        L{Deferred} with C{False}.
        """
    return self._test(self.proto.delete(b'bar'), b'delete bar\r\n', b'NOT FOUND\r\n', False)