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
def test_credentialsWithoutSignature(self):
    """
        Calling L{checkers.SSHPublicKeyChecker.requestAvatarId} with
        credentials that do not have a signature fails with L{ValidPublicKey}.
        """
    self.credentials.signature = None
    self.failureResultOf(self.checker.requestAvatarId(self.credentials), ValidPublicKey)