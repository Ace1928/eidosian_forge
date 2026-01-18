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
def do_QUIT(self):
    """
        Handle a QUIT command.

        Remove any messages marked for deletion, issue a successful response,
        and drop the connection.
        """
    if self.mbox:
        self.mbox.sync()
    self.successResponse()
    self.transport.loseConnection()