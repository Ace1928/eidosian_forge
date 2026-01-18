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
def arg_astring(self, line, final=False):
    """
        Parse an astring from the line, return (arg, rest), possibly
        via a deferred (to handle literals)

        @param line: A line that contains a string literal.
        @type line: L{bytes}

        @param final: Is this the final argument?
        @type final L{bool}

        @return: A 2-tuple containing the parsed argument and any
            trailing data, or a L{Deferred} that fires with that
            2-tuple
        @rtype: L{tuple} of (L{bytes}, L{bytes}) or a L{Deferred}

        """
    line = line.strip()
    if not line:
        raise IllegalClientResponse('Missing argument')
    d = None
    arg, rest = (None, None)
    if line[0:1] == b'"':
        try:
            spam, arg, rest = line.split(b'"', 2)
            rest = rest[1:]
        except ValueError:
            raise IllegalClientResponse('Unmatched quotes')
    elif line[0:1] == b'{':
        if line[-1:] != b'}':
            raise IllegalClientResponse('Malformed literal')
        try:
            size = int(line[1:-1])
        except ValueError:
            raise IllegalClientResponse('Bad literal size: ' + repr(line[1:-1]))
        if final and (not size):
            return (b'', b'')
        d = self._stringLiteral(size)
    else:
        arg = line.split(b' ', 1)
        if len(arg) == 1:
            arg.append(b'')
        arg, rest = arg
    return d or (arg, rest)