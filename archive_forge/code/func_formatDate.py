import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def formatDate(mtime):
    now = time.gmtime()
    info = {'month': _months[mtime.tm_mon], 'day': mtime.tm_mday, 'year': mtime.tm_year, 'hour': mtime.tm_hour, 'minute': mtime.tm_min}
    if now.tm_year != mtime.tm_year:
        return '%(month)s %(day)02d %(year)5d' % info
    else:
        return '%(month)s %(day)02d %(hour)02d:%(minute)02d' % info