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
def esmtpState_ehlo(self, code, resp):
    """
        Send an C{EHLO} to the server.

        If L{heloFallback} is C{True}, and there is no requirement for TLS or
        authentication, the client will fall back to basic SMTP.

        @param code: The server status code from the most recently received
            server message.
        @type code: L{int}

        @param resp: The server status response from the most recently received
            server message.
        @type resp: L{bytes}

        @return: L{None}
        """
    self._expected = SUCCESS
    self._okresponse = self.esmtpState_serverConfig
    self._failresponse = self.esmtpEHLORequired
    if self._tlsMode:
        needTLS = False
    else:
        needTLS = self.requireTransportSecurity
    if self.heloFallback and (not self.requireAuthentication) and (not needTLS):
        self._failresponse = self.smtpState_helo
    self.sendLine(b'EHLO ' + self.identity)