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
def test_failedCWD(self):
    """
        Test a failure in CWD command.

        When the PWD command fails, the returned Deferred should errback
        with L{ftp.CommandFailed}.
        """
    self._testLogin()
    d = self.client.cwd('bar/foo')
    self.assertFailure(d, ftp.CommandFailed)
    self.assertEqual(self.transport.value(), b'CWD bar/foo\r\n')
    self.client.lineReceived(b'550 bar/foo: No such file or directory')
    return d