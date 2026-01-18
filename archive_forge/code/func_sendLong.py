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
def sendLong(self, command, params):
    """
        Send a POP3 command to which a long response is expected.

        @type command: L{bytes}
        @param command: A POP3 command.

        @type params: stringifyable L{object}
        @param params: Command arguments.
        """
    if params:
        if not isinstance(params, bytes):
            params = str(params).encode('utf-8')
        self.sendLine(command + b' ' + params)
    else:
        self.sendLine(command)
    self.command = command
    self.mode = FIRST_LONG