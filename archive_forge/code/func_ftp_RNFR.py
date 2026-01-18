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
def ftp_RNFR(self, fromName):
    self._fromName = fromName
    self.state = self.RENAMING
    return (REQ_FILE_ACTN_PENDING_FURTHER_INFO,)