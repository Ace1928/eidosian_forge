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
def test_getHomeDirectory(self):
    """
        L{ftp.SystemFTPRealm.getHomeDirectory} treats the avatarId passed to it
        as a username in the underlying platform and returns that account's home
        directory.
        """
    user = getpass.getuser()
    import pwd
    expected = pwd.getpwnam(user).pw_dir
    realm = ftp.SystemFTPRealm(self.mktemp())
    home = realm.getHomeDirectory(user)
    self.assertEqual(home, filepath.FilePath(expected))