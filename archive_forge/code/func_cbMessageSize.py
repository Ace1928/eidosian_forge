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
def cbMessageSize(size):
    if not size:
        return defer.fail(_POP3MessageDeleted())
    fileDeferred = defer.maybeDeferred(self.mbox.getMessage, msg)
    fileDeferred.addCallback(lambda fObj: (size, fObj))
    return fileDeferred