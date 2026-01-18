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
def parseTime(s):
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    expr = {'day': '(?P<day>3[0-1]|[1-2]\\d|0[1-9]|[1-9]| [1-9])', 'mon': '(?P<mon>\\w+)', 'year': '(?P<year>\\d\\d\\d\\d)'}
    m = re.match('%(day)s-%(mon)s-%(year)s' % expr, s)
    if not m:
        raise ValueError(f'Cannot parse time string {s!r}')
    d = m.groupdict()
    try:
        d['mon'] = 1 + months.index(d['mon'].lower()) % 12
        d['year'] = int(d['year'])
        d['day'] = int(d['day'])
    except ValueError:
        raise ValueError(f'Cannot parse time string {s!r}')
    else:
        return time.struct_time((d['year'], d['mon'], d['day'], 0, 0, 0, -1, -1, -1))