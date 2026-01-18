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
def test_anonymousLoginNotPermitted(self):
    """
        Verify that without an anonymous checker set up, anonymous login is
        rejected.
        """
    self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user='pass'))
    d = self.clientFactory.login(credentials.Anonymous(), 'BRAINS!')
    self.assertFailure(d, UnhandledCredentials)

    def cleanup(ignore):
        errors = self.flushLoggedErrors(UnhandledCredentials)
        self.assertEqual(len(errors), 1)
    d.addCallback(cleanup)
    self.establishClientAndServer()
    self.pump.flush()
    return d