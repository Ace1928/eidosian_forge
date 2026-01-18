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
def do_STAT(self):
    """
        Handle a STAT command.

        @rtype: L{Deferred <defer.Deferred>}
        @return: A deferred which triggers after the response to the STAT
            command has been issued.
        """
    d = defer.maybeDeferred(self.mbox.listMessages)

    def cbMessages(msgs):
        return self._coiterate(formatStatResponse(msgs))

    def ebMessages(err):
        self.failResponse(err.getErrorMessage())
        log.msg('Unexpected do_STAT failure:')
        log.err(err)
    return self._longOperation(d.addCallbacks(cbMessages, ebMessages))