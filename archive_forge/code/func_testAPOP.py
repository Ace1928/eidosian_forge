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
def testAPOP(self):
    p, t = setUp(greet=False)
    p.dataReceived(b'+OK <challenge string goes here>\r\n')
    d = p.login(b'username', b'password')
    self.assertEqual(t.value(), b'APOP username f34f1e464d0d7927607753129cabe39a\r\n')
    p.dataReceived(b'+OK Welcome!\r\n')
    return d.addCallback(self.assertEqual, b'Welcome!')