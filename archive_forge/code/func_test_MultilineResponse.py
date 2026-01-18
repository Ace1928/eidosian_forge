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
def test_MultilineResponse(self):
    """
        Multiline response
        """
    ftpClient = ftp.FTPClientBasic()
    ftpClient.transport = proto_helpers.StringTransport()
    ftpClient.lineReceived(b'220 Imaginary FTP.')
    deferred = ftpClient.queueStringCommand('BLAH')
    result = []
    deferred.addCallback(result.append)
    deferred.addErrback(self.fail)
    ftpClient.lineReceived(b'210-First line.')
    self.assertEqual([], result)
    ftpClient.lineReceived(b'123-Second line.')
    self.assertEqual([], result)
    ftpClient.lineReceived(b'Just some text.')
    self.assertEqual([], result)
    ftpClient.lineReceived(b'Hi')
    self.assertEqual([], result)
    ftpClient.lineReceived(b'')
    self.assertEqual([], result)
    ftpClient.lineReceived(b'321')
    self.assertEqual([], result)
    ftpClient.lineReceived(b'210 Done.')
    self.assertEqual(['210-First line.', '123-Second line.', 'Just some text.', 'Hi', '', '321', '210 Done.'], result[0])