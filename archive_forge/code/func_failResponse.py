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
def failResponse(self, message=b''):
    """
        Send a response indicating failure.

        @type message: stringifyable L{object}
        @param message: An object whose string representation should be
            included in the response.
        """
    if not isinstance(message, bytes):
        message = str(message).encode('utf-8')
    self.sendLine(b'-ERR ' + message)