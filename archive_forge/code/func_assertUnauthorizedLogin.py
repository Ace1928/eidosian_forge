import os
from base64 import encodebytes
from collections import namedtuple
from io import BytesIO
from typing import Optional
from zope.interface.verify import verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet.defer import Deferred
from twisted.python import util
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
def assertUnauthorizedLogin(self, d):
    """
        Asserts that the L{Deferred} passed in is erred back with an
        L{UnauthorizedLogin} L{Failure}.  This reprsents an invalid login for
        this TestCase.

        NOTE: To work, this method's return value must be returned from the
        test method, or otherwise hooked up to the test machinery.

        @param d: a L{Deferred} from an L{IChecker.requestAvatarId} method.
        @type d: L{Deferred}
        @rtype: L{None}
        """
    self.failureResultOf(d, checkers.UnauthorizedLogin)