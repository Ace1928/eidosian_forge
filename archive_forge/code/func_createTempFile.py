import io
import os
import socket
import stat
from hashlib import md5
from typing import IO
from zope.interface import implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, interfaces, reactor
from twisted.mail import mail, pop3, smtp
from twisted.persisted import dirdbm
from twisted.protocols import basic
from twisted.python import failure, log
def createTempFile(self):
    """
        Create a temporary file to hold the message as it is being transferred.
        """
    attr = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOINHERIT', 0) | getattr(os, 'O_NOFOLLOW', 0)
    tries = 0
    self.fh = -1
    while True:
        self.tmpname = os.path.join(self.mbox.path, 'tmp', _generateMaildirName())
        try:
            self.fh = self.osopen(self.tmpname, attr, 384)
            return None
        except OSError:
            tries += 1
            if tries > 500:
                self.defer.errback(RuntimeError('Could not create tmp file for %s' % self.mbox.path))
                self.defer = None
                return None