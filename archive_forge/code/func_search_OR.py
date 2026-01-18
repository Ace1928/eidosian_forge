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
def search_OR(self, query, id, msg, lastIDs):
    """
        Returns C{True} if the message matches any of the first two query
        items.

        @type query: A L{list} of L{str}
        @param query: A list representing the parsed form of the search query.

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
    a = self._singleSearchStep(query, id, msg, lastSequenceId, lastMessageId)
    b = self._singleSearchStep(query, id, msg, lastSequenceId, lastMessageId)
    return a or b