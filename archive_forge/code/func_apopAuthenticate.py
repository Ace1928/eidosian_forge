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
def apopAuthenticate(self, user, password, magic):
    """
        Perform an authenticated login.

        @type user: L{bytes}
        @param user: The username with which to log in.

        @type password: L{bytes}
        @param password: The password with which to log in.

        @type magic: L{bytes}
        @param magic: The challenge provided by the server.
        """
    digest = md5(magic + password).hexdigest().encode('ascii')
    self.apop(user, digest)