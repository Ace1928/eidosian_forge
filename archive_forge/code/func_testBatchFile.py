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
def testBatchFile(self):
    """
        Test whether batch file function of cftp ('cftp -b batchfile').
        This works by treating the file as a list of commands to be run.
        """
    cmds = 'pwd\nls\nexit\n'

    def _cbCheckResult(res):
        res = res.split(b'\n')
        log.msg('RES %s' % repr(res))
        self.assertIn(self.testDir.asBytesMode().path, res[1])
        self.assertEqual(res[3:-2], [b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1'])
    d = self._getBatchOutput(cmds)
    d.addCallback(_cbCheckResult)
    return d