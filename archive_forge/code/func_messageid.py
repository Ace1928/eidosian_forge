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
def messageid(uniq=None, N=lambda: next(_gen)):
    """
    Return a globally unique random string in RFC 2822 Message-ID format

    <datetime.pid.random@host.dom.ain>

    Optional uniq string will be added to strengthen uniqueness if given.
    """
    datetime = time.strftime('%Y%m%d%H%M%S', time.gmtime())
    pid = os.getpid()
    rand = random.randrange(2 ** 31 - 1)
    if uniq is None:
        uniq = ''
    else:
        uniq = '.' + uniq
    return '<{}.{}.{}{}.{}@{}>'.format(datetime, pid, rand, uniq, N(), DNSNAME.decode()).encode()