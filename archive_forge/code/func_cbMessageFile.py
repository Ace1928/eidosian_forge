import base64
import binascii
import warnings
from hashlib import md5
from typing import Optional
from zope.interface import implementer
from twisted import cred
from twisted.internet import defer, interfaces, task
from twisted.mail import smtp
from twisted.mail._except import POP3ClientError, POP3Error, _POP3MessageDeleted
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.mail._except import (
from twisted.mail._pop3client import POP3Client as AdvancedPOP3Client
def cbMessageFile(info):
    if info is None:
        return
    self._highest = max(self._highest, int(i))
    resp, fp = info
    fp = fpWrapper(fp)
    self.successResponse(successResponse(resp))
    s = basic.FileSender()
    d = s.beginFileTransfer(fp, self.transport, self.transformChunk)

    def cbFileTransfer(lastsent):
        if lastsent != b'\n':
            line = b'\r\n.'
        else:
            line = b'.'
        self.sendLine(line)

    def ebFileTransfer(err):
        self.transport.loseConnection()
        log.msg('Unexpected error in _sendMessageContent:')
        log.err(err)
    d.addCallback(cbFileTransfer)
    d.addErrback(ebFileTransfer)
    return d