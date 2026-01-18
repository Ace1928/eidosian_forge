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
def test_concurrentLogin(self):
    """
        Two different correct login attempts can be made on the same root
        object at the same time and produce two different resulting avatars.
        """
    self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(foo=b'bar', baz=b'quux'))
    firstLogin = self.clientFactory.login(credentials.UsernamePassword(b'foo', b'bar'), 'BRAINS!')
    secondLogin = self.clientFactory.login(credentials.UsernamePassword(b'baz', b'quux'), 'BRAINS!')
    d = gatherResults([firstLogin, secondLogin])

    def cbLoggedIn(result):
        first, second = result
        return gatherResults([first.callRemote('getAvatarId'), second.callRemote('getAvatarId')])
    d.addCallback(cbLoggedIn)

    def cbAvatarIds(x):
        first, second = x
        self.assertEqual(first, b'foo')
        self.assertEqual(second, b'baz')
    d.addCallback(cbAvatarIds)
    self.establishClientAndServer()
    self.pump.flush()
    return d