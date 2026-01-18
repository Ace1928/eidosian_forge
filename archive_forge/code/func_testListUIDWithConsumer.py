import inspect
import sys
from typing import List
from unittest import skipIf
from zope.interface import directlyProvides
import twisted.mail._pop3client
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.mail.pop3 import (
from twisted.mail.test import pop3testserver
from twisted.protocols import basic, loopback
from twisted.python import log
from twisted.trial.unittest import TestCase
def testListUIDWithConsumer(self):
    p, t = setUp()
    c = ListConsumer()
    f = c.consume
    d = p.listUID(f)
    self.assertEqual(t.value(), b'UIDL\r\n')
    p.dataReceived(b'+OK Here it comes\r\n')
    p.dataReceived(b'1 xyz\r\n2 abc\r\n5 mno\r\n')
    self.assertEqual(c.data, {0: [b'xyz'], 1: [b'abc'], 4: [b'mno']})
    p.dataReceived(b'.\r\n')
    return d.addCallback(self.assertIdentical, f)