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
def test_getRootObject(self):
    """
        Assert that L{PBClientFactory.getRootObject}'s Deferred fires with
        a L{RemoteReference}, and that disconnecting it runs its
        disconnection callbacks.
        """
    self.establishClientAndServer()
    rootObjDeferred = self.clientFactory.getRootObject()

    def gotRootObject(rootObj):
        self.assertIsInstance(rootObj, pb.RemoteReference)
        return rootObj

    def disconnect(rootObj):
        disconnectedDeferred = Deferred()
        rootObj.notifyOnDisconnect(disconnectedDeferred.callback)
        self.clientFactory.disconnect()
        self.completeClientLostConnection()
        return disconnectedDeferred
    rootObjDeferred.addCallback(gotRootObject)
    rootObjDeferred.addCallback(disconnect)
    return rootObjDeferred