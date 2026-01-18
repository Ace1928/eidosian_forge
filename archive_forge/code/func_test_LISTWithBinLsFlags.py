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
def test_LISTWithBinLsFlags(self):
    """
        LIST ignores requests for folder with names like '-al' and will list
        the content of current folder.
        """
    os.mkdir(os.path.join(self.directory, 'foo'))
    os.mkdir(os.path.join(self.directory, 'bar'))
    d = self._anonymousLogin()
    self._download('LIST -aL', chainDeferred=d)

    def checkDownload(download):
        names = []
        for line in download.splitlines():
            names.append(line.split(b' ')[-1])
        self.assertEqual(2, len(names))
        self.assertIn(b'foo', names)
        self.assertIn(b'bar', names)
    return d.addCallback(checkDownload)