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
def test_existingRemoteDirectory(self):
    """
        Test that a C{mkdir} on an existing directory fails with the
        appropriate error, and doesn't log an useless error server side.
        """

    def _check(results):
        self.assertEqual(results[0], b'')
        self.assertEqual(results[1], b'remote error 11: mkdir failed')
    d = self.runScript('mkdir testMakeDirectory', 'mkdir testMakeDirectory')
    d.addCallback(_check)
    return d