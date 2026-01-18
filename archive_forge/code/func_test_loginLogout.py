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
def test_loginLogout(self):
    """
        Test that login can be performed with IUsernamePassword credentials and
        that when the connection is dropped the avatar is logged out.
        """
    self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
    creds = credentials.UsernamePassword(b'user', b'pass')
    mind = 'BRAINS!'
    loginCompleted = Deferred()
    d = self.clientFactory.login(creds, mind)

    def cbLogin(perspective):
        self.assertTrue(self.realm.lastPerspective.loggedIn)
        self.assertIsInstance(perspective, pb.RemoteReference)
        return loginCompleted

    def cbDisconnect(ignored):
        self.clientFactory.disconnect()
        self.completeClientLostConnection()
    d.addCallback(cbLogin)
    d.addCallback(cbDisconnect)

    def cbLogout(ignored):
        self.assertTrue(self.realm.lastPerspective.loggedOut)
    d.addCallback(cbLogout)
    self.establishClientAndServer()
    self.pump.flush()
    gc.collect()
    self.pump.flush()
    loginCompleted.callback(None)
    return d