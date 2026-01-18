from __future__ import annotations
import base64
import binascii
import calendar
import math
import os
import re
import tempfile
import time
import warnings
from email import message_from_bytes
from email.message import EmailMessage
from io import BytesIO
from typing import AnyStr, Callable, Dict, List, Optional, Tuple
from urllib.parse import (
from zope.interface import Attribute, Interface, implementer, provider
from incremental import Version
from twisted.internet import address, interfaces, protocol
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IProtocol
from twisted.logger import Logger
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString, networkString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.web._responses import (
from twisted.web.http_headers import Headers, _sanitizeLinearWhitespace
from twisted.web.iweb import IAccessLogFormatter, INonQueuedRequestFactory, IRequest
def datetimeToString(msSinceEpoch=None):
    """
    Convert seconds since epoch to HTTP datetime string.

    @rtype: C{bytes}
    """
    if msSinceEpoch == None:
        msSinceEpoch = time.time()
    year, month, day, hh, mm, ss, wd, y, z = time.gmtime(msSinceEpoch)
    s = networkString('%s, %02d %3s %4d %02d:%02d:%02d GMT' % (weekdayname[wd], day, monthname[month], year, hh, mm, ss))
    return s