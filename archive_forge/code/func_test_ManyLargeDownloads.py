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
def test_ManyLargeDownloads(self):
    """
        Download many large files.

        @return: L{Deferred}
        """
    d = self._anonymousLogin()
    for size in range(100000, 110000, 500):
        with open(os.path.join(self.directory, '%d.txt' % (size,)), 'wb') as fObj:
            fObj.write(b'x' * size)
        self._download('RETR %d.txt' % (size,), chainDeferred=d)

        def checkDownload(download, size=size):
            self.assertEqual(size, len(download))
        d.addCallback(checkDownload)
    return d