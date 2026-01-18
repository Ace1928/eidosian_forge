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
def do_SEARCH(self, tag, charset, query, uid=0):
    sm = ISearchableMailbox(self.mbox, None)
    if sm is not None:
        maybeDeferred(sm.search, query, uid=uid).addCallback(self.__cbSearch, tag, self.mbox, uid).addErrback(self.__ebSearch, tag)
    else:
        s = parseIdList(b'1:*')
        maybeDeferred(self.mbox.fetch, s, uid=uid).addCallback(self.__cbManualSearch, tag, self.mbox, query, uid).addErrback(self.__ebSearch, tag)