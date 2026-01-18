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
def setKnownConsoleSize(self, width, height):
    """
        For the duration of this test, patch C{cftp}'s C{fcntl} module to return
        a fixed width and height.

        @param width: the width in characters
        @type width: L{int}
        @param height: the height in characters
        @type height: L{int}
        """
    import tty

    class FakeFcntl:

        def ioctl(self, fd, opt, mutate):
            if opt != tty.TIOCGWINSZ:
                self.fail('Only window-size queries supported.')
            return struct.pack('4H', height, width, 0, 0)
    self.patch(cftp, 'fcntl', FakeFcntl())