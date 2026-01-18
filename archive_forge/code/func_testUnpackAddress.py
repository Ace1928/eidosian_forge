import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def testUnpackAddress(self):
    p = ident.ProcServerMixin()
    self.assertEqual(p.unpackAddress(_addr1 + ':0277'), ('127.0.0.1', 631))