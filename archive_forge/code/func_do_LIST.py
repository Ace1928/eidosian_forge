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
def do_LIST(self, i=None):
    """
        Handle a LIST command.

        @type i: L{bytes} or L{None}
        @param i: A 1-based message index.

        @rtype: L{Deferred <defer.Deferred>}
        @return: A deferred which triggers after the response to the LIST
            command has been issued.
        """
    if i is None:
        d = defer.maybeDeferred(self.mbox.listMessages)

        def cbMessages(msgs):
            return self._coiterate(formatListResponse(msgs))

        def ebMessages(err):
            self.failResponse(err.getErrorMessage())
            log.msg('Unexpected do_LIST failure:')
            log.err(err)
        return self._longOperation(d.addCallbacks(cbMessages, ebMessages))
    else:
        try:
            i = int(i)
            if i < 1:
                raise ValueError()
        except ValueError:
            if not isinstance(i, bytes):
                i = str(i).encode('utf-8')
            self.failResponse(b'Invalid message-number: ' + i)
        else:
            d = defer.maybeDeferred(self.mbox.listMessages, i - 1)

            def cbMessage(msg):
                self.successResponse(b'%d %d' % (i, msg))

            def ebMessage(err):
                errcls = err.check(ValueError, IndexError)
                if errcls is not None:
                    if errcls is IndexError:
                        warnings.warn('twisted.mail.pop3.IMailbox.listMessages may not raise IndexError for out-of-bounds message numbers: raise ValueError instead.', PendingDeprecationWarning)
                    invalidNum = i
                    if invalidNum and (not isinstance(invalidNum, bytes)):
                        invalidNum = str(invalidNum).encode('utf-8')
                    self.failResponse(b'Invalid message-number: ' + invalidNum)
                else:
                    self.failResponse(err.getErrorMessage())
                    log.msg('Unexpected do_LIST failure:')
                    log.err(err)
            d.addCallbacks(cbMessage, ebMessage)
            return self._longOperation(d)