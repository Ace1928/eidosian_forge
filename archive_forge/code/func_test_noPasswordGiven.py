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
def test_noPasswordGiven(self):
    """
        Passing None as the password avoids sending the PASS command.
        """
    ftpClient = ftp.FTPClientBasic()
    ftpClient.transport = proto_helpers.StringTransport()
    ftpClient.lineReceived(b'220 Welcome to Imaginary FTP.')
    ftpClient.queueLogin('bob', None)
    self.assertEqual(b'USER bob\r\n', ftpClient.transport.value())
    ftpClient.transport.clear()
    ftpClient.lineReceived(b'200 Hello bob.')
    self.assertEqual(b'', ftpClient.transport.value())