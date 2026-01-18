import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def test_NLSTEmpty(self):
    """
        NLST with no argument returns the directory listing for the current
        working directory.
        """
    d = self._anonymousLogin()
    self.dirPath.child('test.txt').touch()
    self.dirPath.child('foo').createDirectory()
    self._download('NLST ', chainDeferred=d)

    def checkDownload(download):
        filenames = download[:-2].split(b'\r\n')
        filenames.sort()
        self.assertEqual([b'foo', b'test.txt'], filenames)
    return d.addCallback(checkDownload)