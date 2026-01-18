import base64
import binascii
import os
import random
import re
import socket
import time
import warnings
from email.utils import parseaddr
from io import BytesIO
from typing import Type
from zope.interface import implementer
from twisted import cred
from twisted.copyright import longversion
from twisted.internet import defer, error, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.interfaces import ISSLTransport, ITLSTransport
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.python.runtime import platform
import codecs
def do_MAIL(self, rest):
    if self._from:
        self.sendCode(503, b'Only one sender per message, please')
        return
    self._to = []
    m = self.mail_re.match(rest)
    if not m:
        self.sendCode(501, b'Syntax error')
        return
    try:
        addr = Address(m.group('path'), self.host)
    except AddressError as e:
        self.sendCode(553, networkString(str(e)))
        return
    validated = defer.maybeDeferred(self.validateFrom, self._helo, addr)
    validated.addCallbacks(self._cbFromValidate, self._ebFromValidate)