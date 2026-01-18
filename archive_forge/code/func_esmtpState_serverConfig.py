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
def esmtpState_serverConfig(self, code, resp):
    """
        Handle a positive response to the I{EHLO} command by parsing the
        capabilities in the server's response and then taking the most
        appropriate next step towards entering a mail transaction.
        """
    items = {}
    for line in resp.splitlines():
        e = line.split(None, 1)
        if len(e) > 1:
            items[e[0]] = e[1]
        else:
            items[e[0]] = None
    self.tryTLS(code, resp, items)