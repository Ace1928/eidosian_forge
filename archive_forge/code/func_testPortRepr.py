import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def testPortRepr(self):
    f = MyServerFactory()
    p = reactor.listenTCP(0, f)
    portNo = str(p.getHost().port)
    self.assertFalse(repr(p).find(portNo) == -1)

    def stoppedListening(ign):
        self.assertFalse(repr(p).find(portNo) != -1)
    d = defer.maybeDeferred(p.stopListening)
    return d.addCallback(stoppedListening)