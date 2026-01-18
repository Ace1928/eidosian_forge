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
def test_loginConnectionRefused(self):
    """
        L{PBClientFactory.login} returns a L{Deferred} which is errbacked
        with the L{ConnectionRefusedError} if the underlying connection is
        refused.
        """
    clientFactory = pb.PBClientFactory()
    loginDeferred = clientFactory.login(credentials.UsernamePassword(b'foo', b'bar'))
    clientFactory.clientConnectionFailed(None, failure.Failure(ConnectionRefusedError('Test simulated refused connection')))
    return self.assertFailure(loginDeferred, ConnectionRefusedError)