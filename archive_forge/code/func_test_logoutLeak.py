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
def test_logoutLeak(self):
    """
        The server does not leak a reference when the client disconnects
        suddenly, even if the cred logout function forms a reference cycle with
        the perspective.
        """
    self.mindRef = None

    def setMindRef(mind):
        self.mindRef = weakref.ref(mind)
    clientBroker, serverBroker, pump = connectedServerAndClient(test=self, realm=LeakyRealm(setMindRef))
    connectionBroken = []
    root = clientBroker.remoteForName('root')
    d = root.callRemote('login', b'guest')

    def cbResponse(x):
        challenge, challenger = x
        mind = SimpleRemote()
        return challenger.callRemote('respond', pb.respond(challenge, b'guest'), mind)
    d.addCallback(cbResponse)

    def connectionLost(_):
        pump.stop()
        connectionBroken.append(1)
        serverBroker.connectionLost(failure.Failure(RuntimeError('boom')))
    d.addCallback(connectionLost)
    pump.flush()
    self.assertEqual(connectionBroken, [1])
    gc.collect()
    self.assertIsNone(self.mindRef())