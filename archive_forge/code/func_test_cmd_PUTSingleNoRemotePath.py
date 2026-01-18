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
def test_cmd_PUTSingleNoRemotePath(self):
    """
        A name based on local path is used when remote path is not
        provided.

        The progress is updated while chunks are transferred.
        """
    content = b'Test\r\nContent'
    localPath = self.makeFile(content=content)
    flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
    remoteName = os.path.join('/', os.path.basename(localPath))
    remoteFile = InMemoryRemoteFile(remoteName)
    self.fakeFilesystem.put(remoteName, flags, defer.succeed(remoteFile))
    self.client.client.options['buffersize'] = 10
    deferred = self.client.cmd_PUT(localPath)
    self.successResultOf(deferred)
    self.assertEqual(content, remoteFile.getvalue())
    self.assertTrue(remoteFile._closed)
    self.checkPutMessage([(localPath, remoteName, ['76% 10.0B', '100% 13.0B', '100% 13.0B'])])