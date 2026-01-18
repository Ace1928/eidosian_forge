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
def openFile(self, filename, flags, attrs):
    """
        Open a file.

        This method returns a L{Deferred} that is called back with an object
        that provides the L{ISFTPFile} interface.

        @type filename: L{bytes}
        @param filename: a string representing the file to open.

        @param flags: an integer of the flags to open the file with, ORed together.
        The flags and their values are listed at the bottom of this file.

        @param attrs: a list of attributes to open the file with.  It is a
        dictionary, consisting of 0 or more keys.  The possible keys are::

            size: the size of the file in bytes
            uid: the user ID of the file as an integer
            gid: the group ID of the file as an integer
            permissions: the permissions of the file with as an integer.
            the bit representation of this field is defined by POSIX.
            atime: the access time of the file as seconds since the epoch.
            mtime: the modification time of the file as seconds since the epoch.
            ext_*: extended attributes.  The server is not required to
            understand this, but it may.

        NOTE: there is no way to indicate text or binary files.  it is up
        to the SFTP client to deal with this.
        """
    data = NS(filename) + struct.pack('!L', flags) + self._packAttributes(attrs)
    d = self._sendRequest(FXP_OPEN, data)
    d.addCallback(self._cbOpenHandle, ClientFile, filename)
    return d