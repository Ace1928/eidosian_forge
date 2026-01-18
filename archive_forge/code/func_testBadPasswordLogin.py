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
def testBadPasswordLogin(self):
    p, t = setUp()
    p.allowInsecureLogin = True
    d = p.login(b'username', b'password')
    self.assertEqual(t.value(), b'USER username\r\n')
    p.dataReceived(b'+OK waiting on you\r\n')
    self.assertEqual(t.value(), b'USER username\r\nPASS password\r\n')
    p.dataReceived(b'-ERR bogus login\r\n')
    return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'bogus login'))