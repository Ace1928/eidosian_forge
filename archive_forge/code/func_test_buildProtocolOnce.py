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
def test_buildProtocolOnce(self):
    """
        A L{ftp.DTPFactory} instance's C{buildProtocol} method can be used once
        to create a L{ftp.DTP} instance.
        """
    protocol = self.factory.buildProtocol(None)
    self.assertIsInstance(protocol, ftp.DTP)
    self.assertIsNone(self.factory.buildProtocol(None))