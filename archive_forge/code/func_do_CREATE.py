import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
def do_CREATE(self, tag, name):
    name = _parseMbox(name)
    try:
        result = self.account.create(name)
    except MailboxException as c:
        self.sendNegativeResponse(tag, networkString(str(c)))
    except BaseException:
        self.sendBadResponse(tag, b'Server error encountered while creating mailbox')
        log.err()
    else:
        if result:
            self.sendPositiveResponse(tag, b'Mailbox created')
        else:
            self.sendNegativeResponse(tag, b'Mailbox not created')