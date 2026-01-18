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
def openForReading(self, path):
    """
        Open C{path} for reading.

        @param path: The path, as a list of segments, to open.
        @type path: C{list} of C{unicode}
        @return: A L{Deferred} is returned that will fire with an object
            implementing L{IReadFile} if the file is successfully opened.  If
            C{path} is a directory, or if an exception is raised while trying
            to open the file, the L{Deferred} will fire with an error.
        """
    p = self._path(path)
    if p.isdir():
        return defer.fail(IsADirectoryError(path))
    try:
        f = p.open('r')
    except OSError as e:
        return errnoToFailure(e.errno, path)
    except BaseException:
        return defer.fail()
    else:
        return defer.succeed(_FileReader(f))