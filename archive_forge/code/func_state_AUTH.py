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
def state_AUTH(self, line):
    """
        Handle received lines for the AUTH state in which an authentication
        challenge response from the client is expected.

        Transition back to the COMMAND state.  Check the credentials and
        complete the authorization process with the L{_cbMailbox}
        callback function on success or the L{_ebMailbox} and L{_ebUnexpected}
        errback functions on failure.

        @type line: L{bytes}
        @param line: The challenge response.
        """
    self.state = 'COMMAND'
    try:
        parts = base64.b64decode(line).split(None, 1)
    except binascii.Error:
        self.failResponse(b'Invalid BASE64 encoding')
    else:
        if len(parts) != 2:
            self.failResponse(b'Invalid AUTH response')
            return
        self._auth.username = parts[0]
        self._auth.response = parts[1]
        d = self.portal.login(self._auth, None, IMailbox)
        d.addCallback(self._cbMailbox, parts[0])
        d.addErrback(self._ebMailbox)
        d.addErrback(self._ebUnexpected)