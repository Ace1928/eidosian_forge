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
def test_ignoresLeadingWhitespaceAndEmptyLines(self):
    """
        L{checkers.readAuthorizedKeyFile} ignores leading whitespace in
        lines, as well as empty lines
        """
    fileobj = BytesIO(b'\n                           # ignore\n                           not ignored\n                           ')
    result = checkers.readAuthorizedKeyFile(fileobj, parseKey=lambda x: x)
    self.assertEqual([b'not ignored'], list(result))