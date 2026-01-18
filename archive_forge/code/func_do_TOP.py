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
def do_TOP(self, i, size):
    """
        Handle a TOP command.

        @type i: L{bytes}
        @param i: A 1-based message index.

        @type size: L{bytes}
        @param size: The number of lines of the message to retrieve.

        @rtype: L{Deferred}
        @return: A deferred which triggers after the response to the TOP
            command has been issued.
        """
    try:
        size = int(size)
        if size < 0:
            raise ValueError
    except ValueError:
        self.failResponse('Bad line count argument')
    else:
        return self._sendMessageContent(i, lambda fp: _HeadersPlusNLines(fp, size), lambda size: 'Top of message follows')