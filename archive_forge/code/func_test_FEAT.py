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
def test_FEAT(self):
    """
        When the server receives 'FEAT', it should report the list of supported
        features. (Additionally, ensure that the server reports various
        particular features that are supported by all Twisted FTP servers.)
        """
    d = self.client.queueStringCommand('FEAT')

    def gotResponse(responseLines):
        self.assertEqual('211-Features:', responseLines[0])
        self.assertIn(' MDTM', responseLines)
        self.assertIn(' PASV', responseLines)
        self.assertIn(' TYPE A;I', responseLines)
        self.assertIn(' SIZE', responseLines)
        self.assertEqual('211 End', responseLines[-1])
    return d.addCallback(gotResponse)