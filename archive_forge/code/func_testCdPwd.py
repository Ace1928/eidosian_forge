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
def testCdPwd(self):
    """
        Test that 'pwd' reports the current remote directory, that 'lpwd'
        reports the current local directory, and that changing to a
        subdirectory then changing to its parent leaves you in the original
        remote directory.
        """
    homeDir = self.testDir
    d = self.runScript('pwd', 'lpwd', 'cd testDirectory', 'cd ..', 'pwd')

    def cmdOutput(output):
        """
            Callback function for handling command output.
            """
        cmds = []
        for cmd in output:
            if isinstance(cmd, bytes):
                cmd = cmd.decode('utf-8')
            cmds.append(cmd)
        return cmds[:3] + cmds[4:]
    d.addCallback(cmdOutput)
    d.addCallback(self.assertEqual, [homeDir.path, os.getcwd(), '', homeDir.path])
    return d