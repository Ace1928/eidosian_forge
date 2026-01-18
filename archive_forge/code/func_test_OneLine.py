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
def test_OneLine(self):
    """
        This example line taken from the docstring for FTPFileListProtocol

        @return: L{Deferred} of command response
        """
    line = '-rw-r--r--   1 root     other        531 Jan 29 03:26 README'

    def check(fileOther):
        (file,), other = fileOther
        self.assertFalse(other, f'unexpect unparsable lines: {repr(other)}')
        self.assertTrue(file['filetype'] == '-', 'misparsed fileitem')
        self.assertTrue(file['perms'] == 'rw-r--r--', 'misparsed perms')
        self.assertTrue(file['owner'] == 'root', 'misparsed fileitem')
        self.assertTrue(file['group'] == 'other', 'misparsed fileitem')
        self.assertTrue(file['size'] == 531, 'misparsed fileitem')
        self.assertTrue(file['date'] == 'Jan 29 03:26', 'misparsed fileitem')
        self.assertTrue(file['filename'] == 'README', 'misparsed fileitem')
        self.assertTrue(file['nlinks'] == 1, 'misparsed nlinks')
        self.assertFalse(file['linktarget'], 'misparsed linktarget')
    return self.getFilesForLines([line]).addCallback(check)