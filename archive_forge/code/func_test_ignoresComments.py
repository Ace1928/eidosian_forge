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
def test_ignoresComments(self):
    """
        L{checkers.readAuthorizedKeyFile} does not attempt to turn comments
        into keys
        """
    fileobj = BytesIO(b'# this comment is ignored\nthis is not\n# this is again\nand this is not')
    result = checkers.readAuthorizedKeyFile(fileobj, lambda x: x)
    self.assertEqual([b'this is not', b'and this is not'], list(result))