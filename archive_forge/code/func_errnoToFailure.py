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
def errnoToFailure(e, path):
    """
    Map C{OSError} and C{IOError} to standard FTP errors.
    """
    if e == errno.ENOENT:
        return defer.fail(FileNotFoundError(path))
    elif e == errno.EACCES or e == errno.EPERM:
        return defer.fail(PermissionDeniedError(path))
    elif e == errno.ENOTDIR:
        return defer.fail(IsNotADirectoryError(path))
    elif e == errno.EEXIST:
        return defer.fail(FileExistsError(path))
    elif e == errno.EISDIR:
        return defer.fail(IsADirectoryError(path))
    else:
        return defer.fail()