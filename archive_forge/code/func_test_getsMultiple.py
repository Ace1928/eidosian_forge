from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_getsMultiple(self):
    """
        L{MemCacheProtocol.getMultiple} handles an additional cas field in the
        returned tuples if C{withIdentifier} is C{True}.
        """
    return self._test(self.proto.getMultiple([b'foo', b'bar'], True), b'gets foo bar\r\n', b'VALUE foo 0 3 1234\r\negg\r\nVALUE bar 0 4 2345\r\nspam\r\nEND\r\n', {b'bar': (0, b'2345', b'spam'), b'foo': (0, b'1234', b'egg')})