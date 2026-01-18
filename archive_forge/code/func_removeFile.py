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
def removeFile(self, filename):
    """
        Remove the given file.

        This method returns a Deferred that is called back when it succeeds.

        @type filename: L{bytes}
        @param filename: the name of the file as a string.
        """
    return self._sendRequest(FXP_REMOVE, NS(filename))