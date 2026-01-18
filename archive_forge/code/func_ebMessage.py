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