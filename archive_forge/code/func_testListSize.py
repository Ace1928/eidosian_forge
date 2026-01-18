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
def testListSize(self):
    p, t = setUp()
    d = p.listSize()
    self.assertEqual(t.value(), b'LIST\r\n')
    p.dataReceived(b'+OK Here it comes\r\n')
    p.dataReceived(b'1 3\r\n2 2\r\n3 1\r\n.\r\n')
    return d.addCallback(self.assertEqual, [3, 2, 1])