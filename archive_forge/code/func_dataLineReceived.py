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
def dataLineReceived(self, line):
    if line[:1] == b'.':
        if line == b'.':
            self.mode = COMMAND
            if self.datafailed:
                self.sendCode(self.datafailed.code, self.datafailed.resp)
                return
            if not self.__messages:
                self._messageHandled('thrown away')
                return
            defer.DeferredList([m.eomReceived() for m in self.__messages], consumeErrors=True).addCallback(self._messageHandled)
            del self.__messages
            return
        line = line[1:]
    if self.datafailed:
        return
    try:
        if not self.__inheader and (not self.__inbody):
            if b':' in line:
                self.__inheader = 1
            elif line:
                for message in self.__messages:
                    message.lineReceived(b'')
                self.__inbody = 1
        if not line:
            self.__inbody = 1
        for message in self.__messages:
            message.lineReceived(line)
    except SMTPServerError as e:
        self.datafailed = e
        for message in self.__messages:
            message.connectionLost()