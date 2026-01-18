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
def collapseStrings(results):
    """
    Turns a list of length-one strings and lists into a list of longer
    strings and lists.  For example,

    ['a', 'b', ['c', 'd']] is returned as ['ab', ['cd']]

    @type results: L{list} of L{bytes} and L{list}
    @param results: The list to be collapsed

    @rtype: L{list} of L{bytes} and L{list}
    @return: A new list which is the collapsed form of C{results}
    """
    copy = []
    begun = None
    pred = lambda e: isinstance(e, tuple)
    tran = {0: lambda e: splitQuoted(b''.join(e)), 1: lambda e: [b''.join([i[0] for i in e])]}
    for i, c in enumerate(results):
        if isinstance(c, list):
            if begun is not None:
                copy.extend(splitOn(results[begun:i], pred, tran))
                begun = None
            copy.append(collapseStrings(c))
        elif begun is None:
            begun = i
    if begun is not None:
        copy.extend(splitOn(results[begun:], pred, tran))
    return copy