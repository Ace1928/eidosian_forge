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
def renameFile(self, oldpath, newpath):
    """
        Rename the given file.

        This method returns a Deferred that is called back when it succeeds.

        @type oldpath: L{bytes}
        @param oldpath: the current location of the file.
        @type newpath: L{bytes}
        @param newpath: the new file name.
        """
    return self._sendRequest(FXP_RENAME, NS(oldpath) + NS(newpath))