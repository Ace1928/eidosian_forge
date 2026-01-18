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
def test_multilineRemoveFileResponse(self):
    """
        If the server returns multiple response lines, the L{Deferred} returned
        by L{ftp.FTPClient.removeFile} is still fired with a true value if the
        ultimate response code is 250.
        """
    self._testLogin()
    d = self.client.removeFile('/tmp/test')
    self.client.lineReceived(b'250-perhaps a progress report')
    self.client.lineReceived(b'250 okay')
    return d.addCallback(self.assertTrue)