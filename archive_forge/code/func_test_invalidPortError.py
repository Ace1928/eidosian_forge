import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def test_invalidPortError(self):
    """
        'INVALID-PORT' error should map to the L{ident.InvalidPort} exception.
        """
    d = defer.Deferred()
    self.client.queries.append((d, 345, 567))
    self.client.lineReceived('345, 567 :  ERROR : INVALID-PORT')
    return self.assertFailure(d, ident.InvalidPort)