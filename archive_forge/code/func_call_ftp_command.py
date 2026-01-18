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
def call_ftp_command(command):
    method = getattr(self, 'ftp_' + command, None)
    if method is not None:
        return method(*params)
    return defer.fail(CmdNotImplementedError(command))