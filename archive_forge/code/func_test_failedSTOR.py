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
def test_failedSTOR(self):
    """
        Test a failure in the STOR command.

        If the server does not acknowledge successful receipt of the
        uploaded file, the second Deferred returned by
        L{ftp.FTPClient.storeFile} should errback with L{ftp.CommandFailed}.
        """
    tr = proto_helpers.StringTransport()

    def cbStore(sender):
        self.client.lineReceived(b'150 File status okay; about to open data connection.')
        sender.transport.write(b'x' * 1000)
        sender.finish()
        sender.connectionLost(failure.Failure(error.ConnectionDone('')))

    def cbConnect(host, port, factory):
        self.assertEqual(host, '127.0.0.1')
        self.assertEqual(port, 12345)
        proto = factory.buildProtocol((host, port))
        proto.makeConnection(tr)
    self.client.connectFactory = cbConnect
    self._testLogin()
    d1, d2 = self.client.storeFile('spam')
    d1.addCallback(cbStore)
    self.assertFailure(d2, ftp.CommandFailed)
    self.assertEqual(self.transport.value(), b'PASV\r\n')
    self.transport.clear()
    self.client.lineReceived(passivemode_msg(self.client))
    self.assertEqual(self.transport.value(), b'STOR spam\r\n')
    self.transport.clear()
    self.client.lineReceived(b'426 Transfer aborted.  Data connection closed.')
    return defer.gatherResults([d1, d2])