import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
def test_defer(self):
    c, s, pump = connectedServerAndClient(test=self)
    d = DeferredRemote()
    s.setNameForLocal('d', d)
    e = c.remoteForName('d')
    pump.pump()
    pump.pump()
    results = []
    e.callRemote('doItLater').addCallback(results.append)
    pump.pump()
    pump.pump()
    self.assertFalse(d.run, 'Deferred method run too early.')
    d.d.callback(5)
    self.assertEqual(d.run, 5, 'Deferred method run too late.')
    pump.pump()
    pump.pump()
    self.assertEqual(results[0], 6, 'Incorrect result.')