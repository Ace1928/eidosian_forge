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
def formatStatResponse(msgs):
    """
    Format a list of message sizes into a STAT response.

    This generator function is intended to be used with
    L{Cooperator <twisted.internet.task.Cooperator>}.

    @type msgs: L{list} of L{int}
    @param msgs: A list of message sizes.

    @rtype: L{None} or L{bytes}
    @return: Yields none until a result is available, then a string that is
        suitable for use in a STAT response. The string consists of the number
        of messages and the total size of the messages in octets.
    """
    i = 0
    bytes = 0
    for size in msgs:
        i += 1
        bytes += size
        yield None
    yield successResponse(b'%d %d' % (i, bytes))