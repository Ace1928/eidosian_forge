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
def test_filenameWithUnescapedSpace(self):
    """
        Will parse filenames and linktargets containing unescaped
        space characters.
        """
    line1 = 'drw-r--r--   2 root     other        531 Jan  9  2003 A B'
    line2 = 'lrw-r--r--   1 root     other          1 Jan 29 03:26 B A -> D C/A B'

    def check(result):
        files, others = result
        self.assertEqual([], others, 'unexpected others entries')
        self.assertEqual('A B', files[0]['filename'], 'misparsed filename')
        self.assertEqual('B A', files[1]['filename'], 'misparsed filename')
        self.assertEqual('D C/A B', files[1]['linktarget'], 'misparsed linktarget')
    return self.getFilesForLines([line1, line2]).addCallback(check)