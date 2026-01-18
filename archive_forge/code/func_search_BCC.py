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
def search_BCC(self, query, id, msg):
    """
        Returns C{True} if the message has a BCC address matching the query.

        @type query: A L{list} of L{str}
        @param query: A list whose first element is a BCC L{str}

        @type id: L{int}
        @param id: The sequence number of the message being checked.

        @type msg: Provider of L{imap4.IMessage}
        """
    bcc = msg.getHeaders(False, 'bcc').get('bcc', '')
    return bcc.lower().find(query.pop(0).lower()) != -1