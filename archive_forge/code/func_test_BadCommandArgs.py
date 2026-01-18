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
def test_BadCommandArgs(self):
    """
        Send command with bad arguments.

        @return: L{Deferred} of command response
        """
    d = self._anonymousLogin()
    self.assertCommandFailed('MODE z', ["504 Not implemented for parameter 'z'."], chainDeferred=d)
    self.assertCommandFailed('STRU I', ["504 Not implemented for parameter 'I'."], chainDeferred=d)
    return d