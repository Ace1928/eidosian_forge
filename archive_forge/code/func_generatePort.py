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
def generatePort(portCmd):
    portCmd.text = 'PORT {}'.format(ftp.encodeHostPort('127.0.0.1', 9876))
    portCmd.protocol.makeConnection(proto_helpers.StringTransport())
    self.client.lineReceived(b'150 File status okay; about to open data connection.')
    portCmd.protocol.dataReceived(b'foo\r\n')
    portCmd.protocol.dataReceived(b'bar\r\n')
    portCmd.protocol.dataReceived(b'baz\r\n')
    portCmd.protocol.connectionLost(failure.Failure(error.ConnectionDone('')))