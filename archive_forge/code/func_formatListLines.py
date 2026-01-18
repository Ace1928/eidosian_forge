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
def formatListLines(msgs):
    """
    Format a list of message sizes for use in a LIST response.

    @type msgs: L{list} of L{int}
    @param msgs: A list of message sizes.

    @rtype: L{bytes}
    @return: Yields a series of strings that are suitable for use as scan
        listings in a LIST response. Each string consists of a message number
        and its size in octets.
    """
    i = 0
    for size in msgs:
        i += 1
        yield (b'%d %d\r\n' % (i, size))