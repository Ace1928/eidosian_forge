from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_emptyGets(self):
    """
        Test getting a non-available key with gets: it succeeds but return
        L{None} as value, C{0} as flag and an empty cas value.
        """
    return self._test(self.proto.get(b'foo', True), b'gets foo\r\n', b'END\r\n', (0, b'', None))