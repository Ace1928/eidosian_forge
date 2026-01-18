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
def testSSLTransportConsideredSecure(self):
    """
        If a server doesn't offer APOP but the transport is secured using
        SSL or TLS, a plaintext login should be allowed, not rejected with
        an InsecureAuthenticationDisallowed exception.
        """
    p, t = setUp(greet=False)
    directlyProvides(t, interfaces.ISSLTransport)
    p.dataReceived(b'+OK Howdy\r\n')
    d = p.login(b'username', b'password')
    self.assertEqual(t.value(), b'USER username\r\n')
    t.clear()
    p.dataReceived(b'+OK\r\n')
    self.assertEqual(t.value(), b'PASS password\r\n')
    p.dataReceived(b'+OK\r\n')
    return d