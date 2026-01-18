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
def do_DATA(self, rest):
    if self._from is None or not self._to:
        self.sendCode(503, b'Must have valid receiver and originator')
        return
    self.mode = DATA
    helo, origin = (self._helo, self._from)
    recipients = self._to
    self._from = None
    self._to = []
    self.datafailed = None
    msgs = []
    for user, msgFunc in recipients:
        try:
            msg = msgFunc()
            rcvdhdr = self.receivedHeader(helo, origin, [user])
            if rcvdhdr:
                msg.lineReceived(rcvdhdr)
            msgs.append(msg)
        except SMTPServerError as e:
            self.sendCode(e.code, e.resp)
            self.mode = COMMAND
            self._disconnect(msgs)
            return
        except BaseException:
            log.err()
            self.sendCode(550, b'Internal server error')
            self.mode = COMMAND
            self._disconnect(msgs)
            return
    self.__messages = msgs
    self.__inheader = self.__inbody = 0
    self.sendCode(354, b'Continue')
    if self.noisy:
        fmt = 'Receiving message for delivery: from=%s to=%s'
        log.msg(fmt % (origin, [str(u) for u, f in recipients]))