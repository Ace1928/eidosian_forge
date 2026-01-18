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
def spewMessage(self, id, msg, query, uid):
    wbuf = WriteBuffer(self.transport)
    write = wbuf.write
    flush = wbuf.flush

    def start():
        write(b'* %d FETCH (' % (id,))

    def finish():
        write(b')\r\n')

    def space():
        write(b' ')

    def spew():
        seenUID = False
        start()
        for part in query:
            if part.type == 'uid':
                seenUID = True
            if part.type == 'body':
                yield self.spew_body(part, id, msg, write, flush)
            else:
                f = getattr(self, 'spew_' + part.type)
                yield f(id, msg, write, flush)
            if part is not query[-1]:
                space()
        if uid and (not seenUID):
            space()
            yield self.spew_uid(id, msg, write, flush)
        finish()
        flush()
    return self._scheduler(spew())