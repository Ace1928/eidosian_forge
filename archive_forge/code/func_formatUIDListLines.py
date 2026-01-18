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
def formatUIDListLines(msgs, getUidl):
    """
    Format a list of message sizes for use in a UIDL response.

    @param msgs: See L{formatUIDListResponse}
    @param getUidl: See L{formatUIDListResponse}

    @rtype: L{bytes}
    @return: Yields a series of strings that are suitable for use as unique-id
        listings in a UIDL response. Each string consists of a message number
        and its unique id.
    """
    for i, m in enumerate(msgs):
        if m is not None:
            uid = getUidl(i)
            if not isinstance(uid, bytes):
                uid = str(uid).encode('utf-8')
            yield (b'%d %b\r\n' % (i + 1, uid))