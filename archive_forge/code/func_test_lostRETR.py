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
def test_lostRETR(self):
    """
        Try a RETR, but disconnect during the transfer.
        L{ftp.FTPClient.retrieveFile} should return a Deferred which
        errbacks with L{ftp.ConnectionLost)
        """
    self.client.passive = False
    l = []

    def generatePort(portCmd):
        portCmd.text = 'PORT {}'.format(ftp.encodeHostPort('127.0.0.1', 9876))
        tr = proto_helpers.StringTransportWithDisconnection()
        portCmd.protocol.makeConnection(tr)
        tr.protocol = portCmd.protocol
        portCmd.protocol.dataReceived(b'x' * 500)
        l.append(tr)
    self.client.generatePortCommand = generatePort
    self._testLogin()
    proto = _BufferingProtocol()
    d = self.client.retrieveFile('spam', proto)
    self.assertEqual(self.transport.value(), 'PORT {}\r\n'.format(ftp.encodeHostPort('127.0.0.1', 9876)).encode(self.client._encoding))
    self.transport.clear()
    self.client.lineReceived(b'200 PORT OK')
    self.assertEqual(self.transport.value(), b'RETR spam\r\n')
    self.assertTrue(l)
    l[0].loseConnection()
    self.transport.loseConnection()
    self.assertFailure(d, ftp.ConnectionLost)
    return d