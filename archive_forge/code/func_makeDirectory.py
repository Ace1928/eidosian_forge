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
def makeDirectory(self, path, attrs):
    """
        Make a directory.

        This method returns a Deferred that is called back when it is
        created.

        @type path: L{bytes}
        @param path: the name of the directory to create as a string.

        @param attrs: a dictionary of attributes to create the directory
        with.  Its meaning is the same as the attrs in the openFile method.
        """
    return self._sendRequest(FXP_MKDIR, NS(path) + self._packAttributes(attrs))