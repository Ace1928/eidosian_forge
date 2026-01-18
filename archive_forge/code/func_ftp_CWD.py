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
def ftp_CWD(self, path):
    try:
        segments = toSegments(self.workingDirectory, path)
    except InvalidPath:
        return defer.fail(FileNotFoundError(path))

    def accessGranted(result):
        self.workingDirectory = segments
        return (REQ_FILE_ACTN_COMPLETED_OK,)
    return self.shell.access(segments).addCallback(accessGranted)