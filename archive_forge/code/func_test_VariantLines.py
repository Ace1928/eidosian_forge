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
def test_VariantLines(self):
    """
        Variant lines.
        """
    line1 = 'drw-r--r--   2 root     other        531 Jan  9  2003 A'
    line2 = 'lrw-r--r--   1 root     other          1 Jan 29 03:26 B -> A'
    line3 = 'woohoo! '

    def check(result):
        (file1, file2), (other,) = result
        self.assertTrue(other == 'woohoo! \r', 'incorrect other line')
        self.assertTrue(file1['filetype'] == 'd', 'misparsed fileitem')
        self.assertTrue(file1['perms'] == 'rw-r--r--', 'misparsed perms')
        self.assertTrue(file1['owner'] == 'root', 'misparsed owner')
        self.assertTrue(file1['group'] == 'other', 'misparsed group')
        self.assertTrue(file1['size'] == 531, 'misparsed size')
        self.assertTrue(file1['date'] == 'Jan  9  2003', 'misparsed date')
        self.assertTrue(file1['filename'] == 'A', 'misparsed filename')
        self.assertTrue(file1['nlinks'] == 2, 'misparsed nlinks')
        self.assertFalse(file1['linktarget'], 'misparsed linktarget')
        self.assertTrue(file2['filetype'] == 'l', 'misparsed fileitem')
        self.assertTrue(file2['perms'] == 'rw-r--r--', 'misparsed perms')
        self.assertTrue(file2['owner'] == 'root', 'misparsed owner')
        self.assertTrue(file2['group'] == 'other', 'misparsed group')
        self.assertTrue(file2['size'] == 1, 'misparsed size')
        self.assertTrue(file2['date'] == 'Jan 29 03:26', 'misparsed date')
        self.assertTrue(file2['filename'] == 'B', 'misparsed filename')
        self.assertTrue(file2['nlinks'] == 1, 'misparsed nlinks')
        self.assertTrue(file2['linktarget'] == 'A', 'misparsed linktarget')
    return self.getFilesForLines([line1, line2, line3]).addCallback(check)