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
def ext_AUTH(self, rest):
    if self.authenticated:
        self.sendCode(503, b'Already authenticated')
        return
    parts = rest.split(None, 1)
    chal = self.challengers.get(parts[0].upper(), lambda: None)()
    if not chal:
        self.sendCode(504, b'Unrecognized authentication type')
        return
    self.mode = AUTH
    self.challenger = chal
    if len(parts) > 1:
        chal.getChallenge()
        rest = parts[1]
    else:
        rest = None
    self.state_AUTH(rest)