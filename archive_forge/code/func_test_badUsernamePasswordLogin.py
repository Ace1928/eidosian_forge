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
def test_badUsernamePasswordLogin(self):
    """
        Test that a login attempt with an invalid user or invalid password
        fails in the appropriate way.
        """
    self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
    firstLogin = self.clientFactory.login(credentials.UsernamePassword(b'nosuchuser', b'pass'))
    secondLogin = self.clientFactory.login(credentials.UsernamePassword(b'user', b'wrongpass'))
    self.assertFailure(firstLogin, UnauthorizedLogin)
    self.assertFailure(secondLogin, UnauthorizedLogin)
    d = gatherResults([firstLogin, secondLogin])

    def cleanup(ignore):
        errors = self.flushLoggedErrors(UnauthorizedLogin)
        self.assertEqual(len(errors), 2)
    d.addCallback(cleanup)
    self.establishClientAndServer()
    self.pump.flush()
    return d