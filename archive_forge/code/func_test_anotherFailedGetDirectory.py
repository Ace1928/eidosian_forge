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
def test_anotherFailedGetDirectory(self):
    """
        Test a different failure in getDirectory method.

        The response should be quoted to be parsed, so it returns an error
        otherwise.
        """
    self._testLogin()
    d = self.client.getDirectory()
    self.assertFailure(d, ftp.CommandFailed)
    self.assertEqual(self.transport.value(), b'PWD\r\n')
    self.client.lineReceived(b'257 /bar/baz')
    return d