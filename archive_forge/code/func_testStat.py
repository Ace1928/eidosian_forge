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
def testStat(self):
    p, t = setUp()
    d = p.stat()
    self.assertEqual(t.value(), b'STAT\r\n')
    p.dataReceived(b'+OK 1 1212\r\n')
    return d.addCallback(self.assertEqual, (1, 1212))