import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
def fetchSpecific(self, messages, uid=0, headerType=None, headerNumber=None, headerArgs=None, peek=None, offset=None, length=None):
    """
        Retrieve a specific section of one or more messages

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
            numbers or of unique message IDs.

        @type headerType: L{str}
        @param headerType: If specified, must be one of HEADER, HEADER.FIELDS,
            HEADER.FIELDS.NOT, MIME, or TEXT, and will determine which part of
            the message is retrieved.  For HEADER.FIELDS and HEADER.FIELDS.NOT,
            C{headerArgs} must be a sequence of header names.  For MIME,
            C{headerNumber} must be specified.

        @type headerNumber: L{int} or L{int} sequence
        @param headerNumber: The nested rfc822 index specifying the entity to
            retrieve.  For example, C{1} retrieves the first entity of the
            message, and C{(2, 1, 3}) retrieves the 3rd entity inside the first
            entity inside the second entity of the message.

        @type headerArgs: A sequence of L{str}
        @param headerArgs: If C{headerType} is HEADER.FIELDS, these are the
            headers to retrieve.  If it is HEADER.FIELDS.NOT, these are the
            headers to exclude from retrieval.

        @type peek: C{bool}
        @param peek: If true, cause the server to not set the \\Seen flag on
            this message as a result of this command.

        @type offset: L{int}
        @param offset: The number of octets at the beginning of the result to
            skip.

        @type length: L{int}
        @param length: The number of octets to retrieve.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a mapping of message
            numbers to retrieved data, or whose errback is invoked if there is
            an error.
        """
    fmt = '%s BODY%s[%s%s%s]%s'
    if headerNumber is None:
        number = ''
    elif isinstance(headerNumber, int):
        number = str(headerNumber)
    else:
        number = '.'.join(map(str, headerNumber))
    if headerType is None:
        header = ''
    elif number:
        header = '.' + headerType
    else:
        header = headerType
    if header and headerType in ('HEADER.FIELDS', 'HEADER.FIELDS.NOT'):
        if headerArgs is not None:
            payload = ' (%s)' % ' '.join(headerArgs)
        else:
            payload = ' ()'
    else:
        payload = ''
    if offset is None:
        extra = ''
    else:
        extra = '<%d.%d>' % (offset, length)
    fetch = uid and b'UID FETCH' or b'FETCH'
    cmd = fmt % (messages, peek and '.PEEK' or '', number, header, payload, extra)
    cmd = cmd.encode('charmap')
    d = self.sendCommand(Command(fetch, cmd, wantResponse=(b'FETCH',)))
    d.addCallback(self._cbFetch, (), False)
    return d