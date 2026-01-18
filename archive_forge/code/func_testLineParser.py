import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def testLineParser(self):
    p = ident.ProcServerMixin()
    self.assertEqual(p.parseLine(self.line), (('127.0.0.1', 25), ('1.2.3.4', 762), 0))