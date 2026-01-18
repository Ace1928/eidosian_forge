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
def test_ignoresNonexistantFile(self):
    """
        L{checkers.UNIXAuthorizedKeysFiles.getAuthorizedKeys} returns only
        the keys in C{~/.ssh/authorized_keys} and C{~/.ssh/authorized_keys2}
        if they exist.
        """
    keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb, parseKey=lambda x: x)
    self.assertEqual(self.expectedKeys, list(keydb.getAuthorizedKeys(b'alice')))