import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def test_lostConnection(self):
    """
        A pending query which failed because of a ConnectionLost should
        receive an L{ident.IdentError}.
        """
    d = defer.Deferred()
    self.client.queries.append((d, 765, 432))
    self.client.connectionLost(failure.Failure(error.ConnectionLost()))
    return self.assertFailure(d, ident.IdentError)