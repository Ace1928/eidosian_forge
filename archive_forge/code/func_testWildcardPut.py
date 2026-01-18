import getpass
import locale
import operator
import os
import struct
import sys
import time
from io import BytesIO, TextIOWrapper
from unittest import skipIf
from zope.interface import implementer
from twisted.conch import ls
from twisted.conch.interfaces import ISFTPFile
from twisted.conch.test.test_filetransfer import FileTransferTestAvatar, SFTPTestBase
from twisted.cred import portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.internet.utils import getProcessOutputAndValue, getProcessValue
from twisted.python import log
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def testWildcardPut(self):
    """
        What happens if you issue a 'put' command and include a wildcard (i.e.
        '*') in parameter? Check that all files matching the wildcard are
        uploaded to the correct directory.
        """

    def check(results):
        self.assertEqual(results[0], b'')
        self.assertEqual(results[2], b'')
        self.assertFilesEqual(self.testDir.child('testRemoveFile'), self.testDir.parent().child('testRemoveFile'), 'testRemoveFile get failed')
        self.assertFilesEqual(self.testDir.child('testRenameFile'), self.testDir.parent().child('testRenameFile'), 'testRenameFile get failed')
    d = self.runScript('cd ..', f'put {self.testDir.path}/testR*', 'cd %s' % self.testDir.basename())
    d.addCallback(check)
    return d