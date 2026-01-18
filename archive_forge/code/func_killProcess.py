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
def killProcess(self):
    """
        Kill the process if it is still running.

        If the process is still running, sends a KILL signal to the transport
        and returns a C{Deferred} which fires when L{processEnded} is called.

        @return: a C{Deferred}.
        """
    if self._processEnded:
        return defer.succeed(None)
    self.onProcessEnd = defer.Deferred()
    self.transport.signalProcess('KILL')
    return self.onProcessEnd