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
def testCapabilityError(self):
    p, t = setUp()
    d = p.capabilities(useCache=0)
    self.assertEqual(t.value(), b'CAPA\r\n')
    p.dataReceived(b'-ERR This server is lame!\r\n')
    return d.addCallback(self.assertEqual, {})