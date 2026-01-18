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
def getEnvelope(msg):
    headers = msg.getHeaders(True)
    date = headers.get('date')
    subject = headers.get('subject')
    from_ = headers.get('from')
    sender = headers.get('sender', from_)
    reply_to = headers.get('reply-to', from_)
    to = headers.get('to')
    cc = headers.get('cc')
    bcc = headers.get('bcc')
    in_reply_to = headers.get('in-reply-to')
    mid = headers.get('message-id')
    return (date, subject, parseAddr(from_), parseAddr(sender), reply_to and parseAddr(reply_to), to and parseAddr(to), cc and parseAddr(cc), bcc and parseAddr(bcc), in_reply_to, mid)