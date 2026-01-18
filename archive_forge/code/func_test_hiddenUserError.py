import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def test_hiddenUserError(self):
    """
        'HIDDEN-USER' error should map to the L{ident.HiddenUser} exception.
        """
    d = defer.Deferred()
    self.client.queries.append((d, 567, 789))
    self.client.lineReceived('567, 789 : ERROR : HIDDEN-USER')
    return self.assertFailure(d, ident.HiddenUser)