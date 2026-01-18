import base64
import binascii
import warnings
from hashlib import md5
from typing import Optional
from zope.interface import implementer
from twisted import cred
from twisted.internet import defer, interfaces, task
from twisted.mail import smtp
from twisted.mail._except import POP3ClientError, POP3Error, _POP3MessageDeleted
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.mail._except import (
from twisted.mail._pop3client import POP3Client as AdvancedPOP3Client
def do_PASS(self, password, *words):
    """
        Handle a PASS command.

        If a USER command was previously received, authenticate the user and
        complete the authorization process with the L{_cbMailbox} callback
        function on success or the L{_ebMailbox} and L{_ebUnexpected} errback
        functions on failure.  If a USER command was not previously received,
        send an error response.

        @type password: L{bytes}
        @param password: A password.

        @type words: L{tuple} of L{bytes}
        @param words: Other parts of the password split by spaces.
        """
    if self._userIs is None:
        self.failResponse(b'USER required before PASS')
        return
    user = self._userIs
    self._userIs = None
    password = b' '.join((password,) + words)
    d = defer.maybeDeferred(self.authenticateUserPASS, user, password)
    d.addCallbacks(self._cbMailbox, self._ebMailbox, callbackArgs=(user,)).addErrback(self._ebUnexpected)