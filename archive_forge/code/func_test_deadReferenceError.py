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
def test_deadReferenceError(self):
    """
        Test that when a connection is lost, calling a method on a
        RemoteReference obtained from it raises L{DeadReferenceError}.
        """
    self.establishClientAndServer()
    rootObjDeferred = self.clientFactory.getRootObject()

    def gotRootObject(rootObj):
        disconnectedDeferred = Deferred()
        rootObj.notifyOnDisconnect(disconnectedDeferred.callback)

        def lostConnection(ign):
            self.assertRaises(pb.DeadReferenceError, rootObj.callRemote, 'method')
        disconnectedDeferred.addCallback(lostConnection)
        self.clientFactory.disconnect()
        self.completeClientLostConnection()
        return disconnectedDeferred
    return rootObjDeferred.addCallback(gotRootObject)