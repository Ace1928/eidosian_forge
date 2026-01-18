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
def collapseNestedLists(items):
    """
    Turn a nested list structure into an s-exp-like string.

    Strings in C{items} will be sent as literals if they contain CR or LF,
    otherwise they will be quoted.  References to None in C{items} will be
    translated to the atom NIL.  Objects with a 'read' attribute will have
    it called on them with no arguments and the returned string will be
    inserted into the output as a literal.  Integers will be converted to
    strings and inserted into the output unquoted.  Instances of
    C{DontQuoteMe} will be converted to strings and inserted into the output
    unquoted.

    This function used to be much nicer, and only quote things that really
    needed to be quoted (and C{DontQuoteMe} did not exist), however, many
    broken IMAP4 clients were unable to deal with this level of sophistication,
    forcing the current behavior to be adopted for practical reasons.

    @type items: Any iterable

    @rtype: L{str}
    """
    pieces = []
    for i in items:
        if isinstance(i, str):
            i = i.encode('ascii')
        if i is None:
            pieces.extend([b' ', b'NIL'])
        elif isinstance(i, int):
            pieces.extend([b' ', networkString(str(i))])
        elif isinstance(i, DontQuoteMe):
            pieces.extend([b' ', i.value])
        elif isinstance(i, bytes):
            if _needsLiteral(i):
                pieces.extend([b' ', b'{%d}' % (len(i),), IMAP4Server.delimiter, i])
            else:
                pieces.extend([b' ', _quote(i)])
        elif hasattr(i, 'read'):
            d = i.read()
            pieces.extend([b' ', b'{%d}' % (len(d),), IMAP4Server.delimiter, d])
        else:
            pieces.extend([b' ', b'(' + collapseNestedLists(i) + b')'])
    return b''.join(pieces[1:])