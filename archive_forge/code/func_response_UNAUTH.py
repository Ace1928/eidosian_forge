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
def response_UNAUTH(self, tag, rest):
    if self.state is None:
        status, rest = rest.split(None, 1)
        if status.upper() == b'OK':
            self.state = 'unauth'
        elif status.upper() == b'PREAUTH':
            self.state = 'auth'
        else:
            self.transport.loseConnection()
            raise IllegalServerResponse(tag + b' ' + rest)
        b, e = (rest.find(b'['), rest.find(b']'))
        if b != -1 and e != -1:
            self.serverGreeting(self.__cbCapabilities(([parseNestedParens(rest[b + 1:e])], None)))
        else:
            self.serverGreeting(None)
    else:
        self._defaultHandler(tag, rest)