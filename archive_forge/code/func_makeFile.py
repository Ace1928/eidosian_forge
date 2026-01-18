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
def makeFile(self, path=None, content=b''):
    """
        Create a local file and return its path.

        When `path` is L{None}, it will create a new temporary file.

        @param path: Optional path for the new file.
        @type path: L{str}

        @param content: Content to be written in the new file.
        @type content: L{bytes}

        @return: Path to the newly create file.
        """
    if path is None:
        path = self.mktemp()
    with open(path, 'wb') as file:
        file.write(content)
    return path