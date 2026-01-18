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
def search_UID(self, query, id, msg, lastIDs):
    """
        Returns C{True} if the message UID is in the range defined by the
        search query.

        @type query: A L{list} of L{bytes}
        @param query: A list representing the parsed form of the search
            query. Its first element should be a L{str} that can be interpreted
            as a sequence range, for example '2:4,5:*'.

        @type id: L{int}
        @param id: The sequence number of the message being checked.

        @type msg: Provider of L{imap4.IMessage}
        @param msg: The message being checked.

        @type lastIDs: L{tuple}
        @param lastIDs: A tuple of (last sequence id, last message id).
        The I{last sequence id} is an L{int} containing the highest sequence
        number of a message in the mailbox.  The I{last message id} is an
        L{int} containing the highest UID of a message in the mailbox.
        """
    lastSequenceId, lastMessageId = lastIDs
    c = query.pop(0)
    m = parseIdList(c, lastMessageId)
    return msg.getUID() in m