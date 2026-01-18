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
def test_RNFRandRNTO(self):
    """
        Sending the RNFR command followed by RNTO, with valid filenames, will
        perform a successful rename operation.
        """
    self.dirPath.child(self.username).createDirectory()
    self.dirPath.child(self.username).child('foo').touch()
    d = self._userLogin()
    self.assertCommandResponse('RNFR foo', ['350 Requested file action pending further information.'], chainDeferred=d)
    self.assertCommandResponse('RNTO bar', ['250 Requested File Action Completed OK'], chainDeferred=d)

    def check_rename(result):
        self.assertTrue(self.dirPath.child(self.username).child('bar').exists())
        return result
    d.addCallback(check_rename)
    return d