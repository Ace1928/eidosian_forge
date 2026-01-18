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
def do_UIDL(self, i=None):
    """
        Handle a UIDL command.

        @type i: L{bytes} or L{None}
        @param i: A 1-based message index.

        @rtype: L{Deferred <defer.Deferred>}
        @return: A deferred which triggers after the response to the UIDL
            command has been issued.
        """
    if i is None:
        d = defer.maybeDeferred(self.mbox.listMessages)

        def cbMessages(msgs):
            return self._coiterate(formatUIDListResponse(msgs, self.mbox.getUidl))

        def ebMessages(err):
            self.failResponse(err.getErrorMessage())
            log.msg('Unexpected do_UIDL failure:')
            log.err(err)
        return self._longOperation(d.addCallbacks(cbMessages, ebMessages))
    else:
        try:
            i = int(i)
            if i < 1:
                raise ValueError()
        except ValueError:
            self.failResponse('Bad message number argument')
        else:
            try:
                msg = self.mbox.getUidl(i - 1)
            except IndexError:
                warnings.warn('twisted.mail.pop3.IMailbox.getUidl may not raise IndexError for out-of-bounds message numbers: raise ValueError instead.', PendingDeprecationWarning)
                self.failResponse('Bad message number argument')
            except ValueError:
                self.failResponse('Bad message number argument')
            else:
                if not isinstance(msg, bytes):
                    msg = str(msg).encode('utf-8')
                self.successResponse(msg)