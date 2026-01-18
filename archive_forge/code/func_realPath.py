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
def realPath(self, path):
    """
        Convert any path to an absolute path.

        This method returns the absolute path as a string, or a Deferred
        that returns the same.

        @type path: L{bytes}
        @param path: the path to convert as a string.
        """
    d = self._sendRequest(FXP_REALPATH, NS(path))
    return d.addCallback(self._cbRealPath)