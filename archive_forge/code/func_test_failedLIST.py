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
def test_failedLIST(self):
    """
        Test a failure in LIST command.

        L{ftp.FTPClient.list} should return a Deferred which fails with
        L{ftp.CommandFailed} if the server indicates the indicated path is
        invalid for some reason.
        """

    def cbConnect(host, port, factory):
        self.assertEqual(host, '127.0.0.1')
        self.assertEqual(port, 12345)
        proto = factory.buildProtocol((host, port))
        proto.makeConnection(proto_helpers.StringTransport())
        self.client.lineReceived(b'150 File status okay; about to open data connection.')
        proto.connectionLost(failure.Failure(error.ConnectionDone('')))
    self.client.connectFactory = cbConnect
    self._testLogin()
    fileList = ftp.FTPFileListProtocol()
    d = self.client.list('foo/bar', fileList)
    self.assertFailure(d, ftp.CommandFailed)
    self.assertEqual(self.transport.value(), b'PASV\r\n')
    self.transport.clear()
    self.client.lineReceived(passivemode_msg(self.client))
    self.assertEqual(self.transport.value(), b'LIST foo/bar\r\n')
    self.client.lineReceived(b'550 foo/bar: No such file or directory')
    return d