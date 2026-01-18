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
def test_ignoresUnparsableKeys(self):
    """
        L{checkers.readAuthorizedKeyFile} does not raise an exception
        when a key fails to parse (raises a
        L{twisted.conch.ssh.keys.BadKeyError}), but rather just keeps going
        """

    def failOnSome(line):
        if line.startswith(b'f'):
            raise keys.BadKeyError('failed to parse')
        return line
    fileobj = BytesIO(b'failed key\ngood key')
    result = checkers.readAuthorizedKeyFile(fileobj, parseKey=failOnSome)
    self.assertEqual([b'good key'], list(result))