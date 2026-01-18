import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
def readLink(self, path):
    """
        Find the root of a set of symbolic links.

        This method returns the target of the link, or a Deferred that
        returns the same.

        @type path: L{bytes}
        @param path: the path of the symlink to read.
        """
    d = self._sendRequest(FXP_READLINK, NS(path))
    return d.addCallback(self._cbRealPath)