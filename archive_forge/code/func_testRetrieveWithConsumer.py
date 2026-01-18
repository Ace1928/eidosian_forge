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
def testRetrieveWithConsumer(self):
    p, t = setUp()
    c = MessageConsumer()
    f = c.consume
    d = p.retrieve(7, f)
    self.assertEqual(t.value(), b'RETR 8\r\n')
    p.dataReceived(b'+OK Message incoming\r\n')
    p.dataReceived(b'La la la here is message text\r\n')
    p.dataReceived(b'..Further message text\r\n.\r\n')
    return d.addCallback(self._cbTestRetrieveWithConsumer, f, c)